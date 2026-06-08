# Sleep / Wake Up API — Design

**Date:** 2026-06-07
**Branch:** `hejian/sleep-wakeup-api`
**Status:** Design (pending review)

## 1. Purpose

Complete TokenSpeed's partially-ported SGLang-style sleep/wake data plane so a
running engine can **release GPU memory** (offload model weights to CPU, discard
KV cache) and later **resume** it, without restarting the process. The primary
driver is the **RL / RLHF training loop**: alternate generation (TokenSpeed) and
weight updates (trainer) on the same GPUs — release frees the card for the
trainer, resume brings the engine back, and fresh weights are pushed via the
existing `update_weights_*` APIs.

This is the **data-plane** complement to the **control-plane** pause/resume API
shipped in PR #346. A "sleep" is the composition of the two: pause + drain the
scheduler (control plane), then release GPU memory (data plane).

## 2. Current state (survey findings)

TokenSpeed is an SGLang-derived codebase and already ships much of the data
plane, but the **back half is unwired and the front half is internally
inconsistent**.

### Already exists

Control plane (fully wired, PR #346):

| Piece | Location |
|---|---|
| Engine API `pause_scheduler` / `resume_scheduler` / `is_scheduler_paused` | `entrypoints/engine.py:303-313` |
| State machine `PauseController`, `PauseState{UNPAUSED, PAUSED_NEW, PAUSED_ALL}`, `scheduler_drained()` | `engine/pause.py` |
| Request dispatch | `engine/request_handler.py:179-186` |
| Event-loop gate (`admit_blocked`, `forward_blocked`, `maybe_finish_drain`, `_paused_idle_step`) | `engine/event_loop.py` |
| Modes `abort` / `wait` / `keep` | `io_struct.py:620` |
| Unit tests | `test/runtime/test_pause_controller.py` |

Data plane (allocator wired, front-end stubbed):

| Piece | Location | Status |
|---|---|---|
| `torch_memory_saver` adapter (`region/pause/resume`, **no tags**) | `utils/torch_memory_saver_adapter.py` | OK |
| `--enable-memory-saver` flag | `server_args.py:256,1604` | OK |
| Weights wrapped in `region()` | `execution/weight_loader.py:86` | OK |
| KV cache wrapped in `region()` | `kv_cache/mha.py:102`, `kv_cache/mla.py:85`, `cache/req_to_token_pool.py:65` | OK |
| Engine API `release_memory_occupation(tags)` / `resume_memory_occupation(tags)` | `entrypoints/engine.py:418-424` | OK |
| Protocol + client communicators + async methods | `protocol.py:189`, `scheduler_control_client.py:161,407-419` | OK |
| io_struct request/response dataclasses | `io_struct.py:735-751` | **empty `pass`** |
| `update_weights_from_{tensor,distributed,disk}` (RL handoff) | `entrypoints/engine.py:359-395` | OK |

### Missing / broken (the work)

1. **Scheduler-side dispatch absent** — `request_handler.py` has no branch for
   `ReleaseMemoryOccupationReqInput` / `ResumeMemoryOccupationReqInput`; the
   client sends but nothing receives.
2. **`memory_saver.pause()/resume()` is never called** in the scheduler process
   (`event_loop.py:172` only constructs the adapters).
3. **Tags bug** — `engine.py:418` builds `ReleaseMemoryOccupationReqInput(tags=tags)`
   but the dataclass (`io_struct.py:735`) is `pass` (no `tags` field) → `TypeError`
   if called today. The adapter's `pause/resume` also take no tags.
4. **No control+data composition** — release must pause + drain *before* freeing
   memory; today the two are disconnected.
5. **No post-wake KV repair**, **no `is_sleeping`**, **no HTTP endpoints**.

## 3. Scope (decided)

- **Scope:** complete the existing SGLang `release/resume_memory_occupation`
  port. Not a from-scratch vLLM-parity `sleep(level, mode)` API.
- **Driver:** RL / RLHF loop.
- **Pause coupling:** auto-pause + drain *inside* release (one call = safe).
- **Tag granularity:** selective `weights` + `kv_cache` tags (enables the RL
  multi-stage wake).
- **Surface:** Python/engine API **and** HTTP endpoints.

## 4. Architecture

Spans the same 7 layers as the pause API, completing the unwired back half.

| Layer | File | Change |
|---|---|---|
| HTTP | `entrypoints/http_server.py` | + `/release_memory_occupation`, `/resume_memory_occupation`, `/is_sleeping` |
| Engine API | `entrypoints/engine.py:418-424` | fix to pass `tags`; add `is_sleeping()` |
| Client / protocol | `engine/scheduler_control_client.py`, `protocol.py` | return outputs; add `is_sleeping` |
| io_struct | `engine/io_struct.py:735-751` | + `tags` on inputs, `success`/`is_sleeping` on outputs |
| Dispatch | `engine/request_handler.py` | + Release / Resume / IsSleeping branches |
| Orchestration | `engine/pause.py` (generalized) + **new `engine/memory_occupation.py`** | core logic |
| Memory adapter | `utils/torch_memory_saver_adapter.py` | + tag pass-through |
| Tag sites | `weight_loader.py`, `kv_cache/{mha,mla}.py`, `cache/req_to_token_pool.py`, `kv_cache/deepseek_v4.py` | + `region(tag=...)` |

### Two new units, each one job

- **`memory_occupation.py`** — pure GPU-memory orchestration: maps
  `tags → memory_saver.pause/resume(tag)`, runs post-wake KV repair, tracks
  `released_tags`. Knows nothing about scheduling or ZMQ.
- **`PauseController` (generalized)** — gains a generic *post-drain action* hook
  so "release after the scheduler drains" reuses the exact deferral that powers
  deferred pause replies. Stays purely about scheduling.

`request_handler` wires them: a release request starts a `wait`-pause and arms a
post-drain action that calls into `memory_occupation`, then replies.

## 5. Core mechanism: generalized deferral

Draining is **asynchronous** — it completes over event-loop iterations via
`maybe_finish_drain()` when `scheduler_drained()` becomes true. Release must run
`memory_saver.pause()` only *after* drain, so release is the same shape as a
deferred pause reply: a post-drain action.

Today `PauseController` holds `_pending_reply: PauseSchedulerReqOutput | None`.
Generalize that single slot into a pending-drain object carrying **two
callbacks**:

```python
@dataclass
class _PendingDrain:
    on_drained:   Callable[[], None]  # scheduler empty → do work + send success
    on_cancelled: Callable[[], None]  # resume arrived first → send failure
```

- **Plain pause** (`abort`/`wait`): `on_drained` sends
  `PauseSchedulerReqOutput(success=True)`; `on_cancelled` sends
  `PauseSchedulerReqOutput(success=False, "resumed before pause drained")`.
  Behavior identical to today.
- **Release**: `on_drained` = `memory_occupation.release(tags)` then send
  `ReleaseMemoryOccupationReqOutput(success=True)`; `on_cancelled` = send
  `ReleaseMemoryOccupationReqOutput(success=False, "resumed before release drained")`.

The existing "a pause is already in progress" guard now naturally rejects a
release while any pause/release is mid-drain (single-consumer promise
preserved). `maybe_finish_drain` runs `on_drained`; `handle_resume`'s pre-drain
path runs `on_cancelled`. Carrying both callbacks routes the wake to the correct
ZMQ communicator (pause vs release — different channels).

## 6. Data flow

### Release `release_memory_occupation(tags=["weights","kv_cache"])`

1. Engine API → client communicator → `ReleaseMemoryOccupationReqInput(tags)`
   fans out to every DP rank's `request_handler`.
2. Handler starts a `wait`-mode pause (`PAUSED_NEW`, cancel grammar-queued) and
   arms `_PendingDrain(on_drained=release, on_cancelled=fail)`.
3. Event loop keeps stepping in-flight requests (`PAUSED_NEW` allows forward)
   until `scheduler_drained()`. New requests buffer.
4. On drain → `on_drained`: `_reset_caches()` (prefix cache is stale),
   `memory_occupation.release(tags)` → `memory_saver.pause(tag)` per tag
   (weights→CPU, KV discarded) → **state to `PAUSED_ALL` + set `released=True`**
   → send `ReleaseMemoryOccupationReqOutput(success=True)`.

### Resume `resume_memory_occupation(tags)` — no drain needed, runs synchronously

1. `memory_occupation.resume(tags)` → `memory_saver.resume(tag)` per tag
   (re-map + restore weights from CPU; re-map KV pages).
2. If `"kv_cache"` resumed → KV repair (§7).
3. If *all* memory now resumed → clear `released`, set `UNPAUSED` (buffered specs
   flush on next admission pass). If partial → stay frozen.
4. Send `ResumeMemoryOccupationReqOutput(success=True)`.

### RL multi-stage (target flow), all while frozen

```
release(["weights","kv_cache"])      # GPU freed for trainer
  → trainer steps on the GPU
resume(["weights"])                  # weights remapped (stale), still frozen
update_weights_from_tensor(...)      # fresh weights overwrite on GPU
resume(["kv_cache"])                 # KV remapped + repaired → UNPAUSED, serving resumes
```

### Correctness: DP idle-forward after release

With data parallelism the event loop's idle step calls `execute_idle_forward()`
to keep ranks in NCCL lockstep — and an idle forward *runs the model*, touching
weights. After `release(["weights"])` those weights are unmapped → idle forward
would crash or read garbage. Because release fans out to **all** DP ranks
together, every rank enters `released` simultaneously, so the fix is consistent:
while `released`, `_paused_idle_step` keeps the lightweight DP *sync* (small
allreduce, no weights) but **skips `execute_idle_forward`**. This is why the
controller needs a `released` flag distinct from `forward_blocked` —
`PAUSED_ALL` alone still permits idle forwards.

## 7. Tag wiring & KV repair

### Tag the existing `region()` call sites

- `weight_loader.py:86` → `region(tag="weights")`
- `kv_cache/mha.py:102`, `kv_cache/mla.py:85` → `region(tag="kv_cache")`
- `cache/req_to_token_pool.py:65` → `region(tag="kv_cache")` (per-token page
  state; invalid once KV discarded)
- `kv_cache/deepseek_v4.py` KV path (currently `del enable_memory_saver`, not
  wrapped) → wrap in `region(tag="kv_cache")` for parity *(verify this path)*

### Adapter tag pass-through

`region(tag=None)`, `pause(tag=None)`, `resume(tag=None)` forward to
`_primary_memory_saver`. The no-op adapter ignores tags. **Requires a
tag-capable `torch_memory_saver`** (optional, unpinned dep today → pin a minimum
version).

### KV repair after wake

Remapped KV pages hold garbage. TokenSpeed loads static FP8 KV scales from the
checkpoint into attention modules — those live in the **weights** tag, so they
ride back with `resume(["weights"])`. Repair is therefore simpler than vLLM's:
**zero the KV buffers** when `kv_cache` is resumed (optional for non-quantized KV
since paging overwrites; removes garbage for FP8). No scale reset needed
*(confirm scale storage location during impl)*.

## 8. API surface

### io_struct (`io_struct.py:735-751`)

```python
ReleaseMemoryOccupationReqInput(tags: list[str] | None = None)
ReleaseMemoryOccupationReqOutput(success: bool = True, message: str = "")
ResumeMemoryOccupationReqInput(tags: list[str] | None = None)
ResumeMemoryOccupationReqOutput(success: bool = True, message: str = "")
IsSleepingReqInput()                       # new
IsSleepingReqOutput(is_sleeping: bool)     # new
```

`tags=None` ⇒ both; otherwise a subset of `{"weights","kv_cache"}`.

### Client / protocol / engine

- `scheduler_control_client.py`: release/resume communicators exist — change the
  two async methods to **return the output** (callers see `success`); add an
  `is_sleeping` communicator + method.
- `protocol.py`: add `is_sleeping` stub.
- `engine.py:418-424`: already passes `tags` (works once io_struct has the
  field); add `is_sleeping()`.
- `engine_base.py:75-79`: widen abstract signatures to take `tags`.

### HTTP (`http_server.py`)

Three thin routes following the existing direct-engine-call pattern (like
`/abort`):

```
POST /release_memory_occupation?tags=weights&tags=kv_cache
POST /resume_memory_occupation?tags=weights
GET  /is_sleeping  → {"is_sleeping": bool}
```

Parse query params, forward to the engine client. *(Check for a dev-mode gate
like vLLM's `VLLM_SERVER_DEV_MODE`; gate if one exists.)*

### `is_sleeping` semantics

`is_sleeping == controller.released` (data-plane: memory is freed). Control-plane
pause is already reported by `is_scheduler_paused`. Documented so callers don't
conflate the two.

## 9. Error handling & edge cases

- **Per-tag bookkeeping** (mirrors vLLM's `sleeping_tags`): `memory_occupation`
  tracks `released_tags: set`. Re-releasing an already-released tag, or resuming
  a not-released tag → `success=False` with message (no double
  `memory_saver.pause`).
- **Memory saver disabled / not installed** (`enable_memory_saver=False` or
  import missing) → release returns `success=False, "memory saver not enabled"`
  rather than silently pausing.
- **Prefix-cache reset on release**: KV is discarded, so release calls the
  existing `_reset_caches()` (matches the pause `clear_cache` contract).
- **resume before release drained** → `on_cancelled` fires `success=False`.
- **pause/release already in progress** → existing single-consumer guard rejects
  the second.
- **`wait` never drains** (a stuck request) → release never completes;
  documented limitation, caller ensures the rollout is idle. A `mode` param is a
  later enhancement.
- **DP aggregate failure**: communicator waits for all rank replies; any rank
  failure → aggregate `success=False`.

## 10. Testing

- **Unit** `test/runtime/test_memory_occupation_controller.py` (mirrors
  `test_pause_controller.py`): mock adapter (records `pause/resume(tag)`), mock
  scheduler (drain predicate), mock send. Covers: deferred release fires on
  drain; `on_cancelled` on resume-before-drain; `released` flag transitions;
  partial-tag resume; `released_tags` rejection; disabled-saver rejection.
- **io_struct / engine signature** round-trip tests.
- **HTTP** route tests (FastAPI testclient): tag parsing + forwarding.
- **GPU integration** (nv2 B200, gated like existing GPU tests): real
  `torch_memory_saver` — `mem_get_info` shows freed/restored; RL multi-stage
  produces coherent output; TP/DP run verifies idle-forward suppression doesn't
  hang NCCL.

## 11. Dependencies, risks, out of scope

### Dependency

- Tag-capable `torch_memory_saver` — pin a minimum version; verify the
  `region/pause/resume(tag=...)` shape on nv2 early (cannot introspect on the dev
  Mac; not installed there).

### Risks to verify on GPU

- (a) exact tag API shape of the installed `torch_memory_saver`;
- (b) `deepseek_v4.py` KV path isn't wrapped in a region today — needs wrapping;
- (c) FP8 KV scale storage location (assumed to ride with weights);
- (d) **CUDA graphs** — remap restores the *same* virtual address so captured
  graphs stay valid by design, but must be tested (TokenSpeed uses cudagraphs
  heavily);
- (e) NCCL lockstep while `released`.

### Out of scope

- vLLM "level 2" true weight-discard with named-buffer snapshot (our path
  offloads weights to CPU = level-1-style; the RL flow's `resume(weights)` +
  `update_weights` is correct, just restores-then-overwrites — a skip-CPU-backup
  optimization is deferred).
- Level-0 pure pause (already shipped as the pause API).
- Auto-wake triggers (timer / queue-threshold).

## 12. File-change summary

New:
- `python/tokenspeed/runtime/engine/memory_occupation.py`
- `test/runtime/test_memory_occupation_controller.py`

Changed:
- `engine/io_struct.py` — add fields + `IsSleeping*` types
- `engine/pause.py` — generalize deferral to `_PendingDrain`; add `released` flag
- `engine/request_handler.py` — Release / Resume / IsSleeping dispatch
- `engine/event_loop.py` — skip `execute_idle_forward` while `released`; wire
  release post-drain action
- `engine/scheduler_control_client.py` — return outputs; `is_sleeping` method
- `engine/protocol.py` — `is_sleeping` stub
- `entrypoints/engine.py` — `is_sleeping()`; (release/resume already pass tags)
- `entrypoints/engine_base.py` — widen signatures
- `entrypoints/http_server.py` — three routes
- `utils/torch_memory_saver_adapter.py` — tag pass-through
- `execution/weight_loader.py`, `layers/attention/kv_cache/{mha,mla,deepseek_v4}.py`,
  `cache/req_to_token_pool.py` — `region(tag=...)`
- dependency pin for `torch_memory_saver`
