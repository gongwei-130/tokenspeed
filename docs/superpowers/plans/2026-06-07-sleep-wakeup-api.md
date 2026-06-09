# Sleep / Wake Up API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete TokenSpeed's SGLang-style data-plane sleep/wake — wire `release_memory_occupation` / `resume_memory_occupation` / `is_sleeping` end-to-end with selective `weights`/`kv_cache` tags, auto-pausing+draining the scheduler before freeing GPU memory, driven by the RL/RLHF loop.

**Architecture:** Reuse the control-plane pause/drain machinery (PR #346). Generalize `PauseController`'s deferred-reply slot into a post-drain *action*, so "release after drain" reuses the proven async-drain path. A new `MemoryOccupationController` owns the GPU-memory orchestration (tag→`memory_saver.pause/resume`, `released_tags` bookkeeping, KV repair) and is driven by `request_handler` dispatch, mirroring how `PauseController` is driven. Pure-Python: the C++ scheduler `.so` is untouched.

**Tech Stack:** Python, `torch_memory_saver` (tag-capable, CUDA virtual-memory unmap/remap), ZMQ control RPC, pytest, nv2 B200 for GPU integration.

**Spec:** `docs/superpowers/specs/2026-06-07-sleep-wakeup-api-design.md`

---

## File structure

New:
- `python/tokenspeed/runtime/engine/memory_occupation.py` — `MemoryOccupationController`: GPU-memory orchestration + `released_tags` + tag validation. No ZMQ knowledge beyond a send func; no scheduling logic beyond delegating to `PauseController`.
- `test/runtime/test_memory_occupation_controller.py` — unit tests (mocked adapter + scheduler + send).
- `test/runtime/test_sleep_wakeup_gpu.py` — GPU-gated integration cases.

Modified:
- `engine/io_struct.py` — add `tags` to inputs, `success`/`message` to outputs, new `IsSleeping*`.
- `engine/pause.py` — generalize `_pending_reply` → `_PendingDrain`; add `released` flag + `request_drain`/`set_released` API.
- `engine/request_handler.py` — dispatch Release/Resume/IsSleeping.
- `engine/event_loop.py` — construct `MemoryOccupationController`, pass to handler, skip `execute_idle_forward` while `released`.
- `engine/scheduler_control_client.py` — return outputs from release/resume; add `is_sleeping`.
- `engine/protocol.py` — add `is_sleeping` stub; widen release/resume.
- `entrypoints/engine.py` — add `is_sleeping()`.
- `entrypoints/engine_base.py` — widen abstract signatures (`tags`).
- `entrypoints/http_server.py` — three routes.
- `utils/torch_memory_saver_adapter.py` — tag pass-through.
- `execution/weight_loader.py`, `layers/attention/kv_cache/{mha,mla,deepseek_v4}.py`, `cache/req_to_token_pool.py` — `region(tag=...)`.
- `python/pyproject.toml` — pin `torch_memory_saver`.

---

## Phase 0 — nv2 feasibility gate (COMPLETED 2026-06-08 on `v2`/b300, 8×B300 SXM6, cu130/torch-2.11)

**Findings (recorded; these refine Tasks 2, 6, 10, 11, 12):**

- **Package:** `torch_memory_saver==0.0.9.post1` — single `manylinux2014` wheel (`cp39-abi3`) that **bundles** the cu13 binaries (`torch_memory_saver_hook_mode_preload_cu13.abi3.so`, `..._torch_cu13.abi3.so`) at the site-packages root. Installs clean via `pip install --break-system-packages` (no nvcc needed). `_detect_cuda_major()` → 13. **NOT preinstalled in the runner image** (`lightseekorg/tokenspeed-runner:latest`) → must be added (image build or runtime install); pin in Task 12.
- **Tag API (confirmed):** `region(tag: str = 'default', enable_cpu_backup: bool = False)`, `pause(tag=None)`, `resume(tag=None)`. The `enable_cpu_backup` flag on `region()` is the offload-vs-discard knob: **weights → `enable_cpu_backup=True`** (byte-exact CPU restore), **kv_cache → `enable_cpu_backup=False`** (discard). This replaces vLLM's pause-time `offload_tags`.
- **Functional proof on B300:** alloc 2 GiB (1 weights + 1 kv) → `pause` freed exactly 2.00 GiB → `resume` restored, weights **byte-exact** → partial wake (weights only) correctly left KV freed (1.10 GiB used). Mechanism works on this driver/torch stack.
- **Hook mode:** **`preload` is required** — it is the only mode that supports pauseable CUDA graphs (entrypoint.py:96; TokenSpeed uses cudagraphs). It needs `LD_PRELOAD` = the preload `.so`, set via `configure_subprocess()`. **Already wired** at `entrypoints/engine.py:528` (non-DP path wraps `proc.start()`); **verify the DP path** (`data_parallel_controller.py`) propagates it (Task 14 Case G).
- **FP8 KV scales:** live on the attention `layer` (`layer.k_scale`, `layer.v_scale`, `layer.k_scale_float`) — i.e. in the **weights** region, restored by `resume(["weights"])`. So KV repair = **zero the KV buffer only**; no scale reset (confirms Task 6 `_kv_repair_after_wake`).
- **deepseek_v4:** `kv_cache/deepseek_v4.py:~793` does `del enable_memory_saver` and never wraps its KV allocation — the V4 model needs explicit region wrapping at its real buffer-alloc site (Task 11). Initial validation uses a small MHA model (e.g. Qwen2-1.5B) where `mha.py` wrapping suffices.

### Task 0: (DONE) Verify torch_memory_saver tag API + KV scale location on nv2

**Files:** none (investigation; record findings in the plan/PR).

- [ ] **Step 1: Reach nv2 and locate the venv**

Run (per memory `nv2-gpu-access`): SSH to b200-80 via the AWS bastion ProxyJump. Activate the project venv.

- [ ] **Step 2: Introspect the installed torch_memory_saver**

Run:
```bash
python -c "import torch_memory_saver as t, inspect; \
print('ver', getattr(t,'__version__','?')); \
s=t.TorchMemorySaver(); \
print('region', inspect.signature(s.region)); \
print('pause ', inspect.signature(s.pause)); \
print('resume', inspect.signature(s.resume))"
```
Expected: `region/pause/resume` accept a `tag` keyword. **If they do not**, the adapter must instead create one `TorchMemorySaver` instance per tag (fallback noted in Task 2); record which path applies.

- [ ] **Step 3: Confirm FP8 KV scale storage**

Run a grep in the model code for `k_scale`/`v_scale`/`load_kv_cache_scales` and confirm scales are attributes of attention modules (i.e. ride with the `weights` region), not in the KV buffer. Record the attribute path.

- [ ] **Step 4: Confirm deepseek_v4 KV allocation site**

Read `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` around line 767-791 (currently `del enable_memory_saver`). Identify where its KV buffers are allocated so Task 13 can wrap them in `region(tag="kv_cache")`.

- [ ] **Step 5: Record findings**

Note the tag-API shape, scale location, and deepseek_v4 site in the PR description. These gate Tasks 2, 12, 13.

---

## Phase 1 — Foundational types & adapter

### Task 1: io_struct fields + IsSleeping types

**Files:**
- Modify: `python/tokenspeed/runtime/engine/io_struct.py:735-751`
- Test: `test/runtime/test_io_struct_memory_occupation.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# test/runtime/test_io_struct_memory_occupation.py
from tokenspeed.runtime.engine.io_struct import (
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    IsSleepingReqInput,
    IsSleepingReqOutput,
)


def test_release_input_defaults_to_all_tags():
    assert ReleaseMemoryOccupationReqInput().tags is None
    assert ReleaseMemoryOccupationReqInput(tags=["weights"]).tags == ["weights"]


def test_resume_input_carries_tags():
    assert ResumeMemoryOccupationReqInput(tags=["kv_cache"]).tags == ["kv_cache"]


def test_outputs_default_success_true():
    assert ReleaseMemoryOccupationReqOutput().success is True
    assert ResumeMemoryOccupationReqOutput().success is True
    assert ReleaseMemoryOccupationReqOutput(success=False, message="x").message == "x"


def test_is_sleeping_output():
    assert IsSleepingReqInput() is not None
    assert IsSleepingReqOutput(is_sleeping=True).is_sleeping is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/runtime/test_io_struct_memory_occupation.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'tags'` and `ImportError` for `IsSleeping*`.

- [ ] **Step 3: Edit io_struct.py**

Replace the four empty dataclasses at lines 735-751 and append the two new types:
```python
@dataclass
class ReleaseMemoryOccupationReqInput:
    tags: list[str] | None = None


@dataclass
class ReleaseMemoryOccupationReqOutput:
    success: bool = True
    message: str = ""


@dataclass
class ResumeMemoryOccupationReqInput:
    tags: list[str] | None = None


@dataclass
class ResumeMemoryOccupationReqOutput:
    success: bool = True
    message: str = ""


@dataclass
class IsSleepingReqInput:
    pass


@dataclass
class IsSleepingReqOutput:
    is_sleeping: bool
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/runtime/test_io_struct_memory_occupation.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add python/tokenspeed/runtime/engine/io_struct.py test/runtime/test_io_struct_memory_occupation.py
git commit -s -m "feat(io_struct): add tags + success to memory-occupation reqs, IsSleeping types"
```

### Task 2: torch_memory_saver_adapter tag pass-through

**Files:**
- Modify: `python/tokenspeed/runtime/utils/torch_memory_saver_adapter.py`
- Test: `test/runtime/test_memory_saver_adapter.py` (create)

- [ ] **Step 1: Write the failing test** (noop adapter must accept + ignore tags)

```python
# test/runtime/test_memory_saver_adapter.py
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


def test_noop_adapter_accepts_tag_kwarg():
    a = TorchMemorySaverAdapter.create(enable=False)
    with a.region(tag="weights", enable_cpu_backup=True):
        pass
    with a.region(tag="kv_cache", enable_cpu_backup=False):
        pass
    a.pause(tag="weights")   # no-op, must not raise
    a.resume(tag="weights")  # no-op, must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/runtime/test_memory_saver_adapter.py -v`
Expected: FAIL — `region()/pause()/resume()` take no `tag` argument.

- [ ] **Step 3: Add tag pass-through**

In `torch_memory_saver_adapter.py`, update the ABC and both impls. Replace the bodies:
```python
class TorchMemorySaverAdapter(ABC):
    @staticmethod
    def create(enable: bool):
        return (
            _TorchMemorySaverAdapterReal() if enable else _TorchMemorySaverAdapterNoop()
        )

    def configure_subprocess(self):
        raise NotImplementedError

    def region(self, tag: str | None = None, enable_cpu_backup: bool = False):
        raise NotImplementedError

    def pause(self, tag: str | None = None):
        raise NotImplementedError

    def resume(self, tag: str | None = None):
        raise NotImplementedError


class _TorchMemorySaverAdapterReal(TorchMemorySaverAdapter):
    def configure_subprocess(self):
        return torch_memory_saver.configure_subprocess()

    def region(self, tag: str | None = None, enable_cpu_backup: bool = False):
        # tag defaults to "default" in the lib; pass through explicitly.
        return _primary_memory_saver.region(
            tag=tag or "default", enable_cpu_backup=enable_cpu_backup
        )

    def pause(self, tag: str | None = None):
        return _primary_memory_saver.pause(tag=tag)

    def resume(self, tag: str | None = None):
        return _primary_memory_saver.resume(tag=tag)


class _TorchMemorySaverAdapterNoop(TorchMemorySaverAdapter):
    @contextmanager
    def configure_subprocess(self):
        yield

    @contextmanager
    def region(self, tag: str | None = None, enable_cpu_backup: bool = False):
        yield

    def pause(self, tag: str | None = None):
        pass

    def resume(self, tag: str | None = None):
        pass
```

> **Confirmed on nv2 (Task 0):** `torch_memory_saver==0.0.9.post1` exposes exactly `region(tag, enable_cpu_backup)`, `pause(tag)`, `resume(tag)`. `enable_cpu_backup=True` → contents restored byte-exact on resume (weights); `False` → discarded (kv_cache).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/runtime/test_memory_saver_adapter.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/tokenspeed/runtime/utils/torch_memory_saver_adapter.py test/runtime/test_memory_saver_adapter.py
git commit -s -m "feat(memory-saver): thread tag through region/pause/resume adapter"
```

---

## Phase 2 — PauseController generalization

### Task 3: Generalize `_pending_reply` → `_PendingDrain` (no behavior change)

**Files:**
- Modify: `python/tokenspeed/runtime/engine/pause.py`
- Test: `test/runtime/test_pause_controller.py` (must stay green)

- [ ] **Step 1: Run the existing suite to capture the baseline**

Run: `pytest test/runtime/test_pause_controller.py -v`
Expected: PASS (baseline before refactor).

- [ ] **Step 2: Introduce `_PendingDrain` and `request_drain`/`set_released`**

Edit `pause.py`. Add the dataclass + imports near the top:
```python
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class _PendingDrain:
    """A deferred action resolved when the scheduler drains.

    ``on_drained`` runs once ``scheduler_drained`` is true (sends the success
    reply, and for a release also frees GPU memory). ``on_cancelled`` runs if a
    resume arrives first (sends the failure reply to the correct communicator).
    """

    on_drained: Callable[[], None]
    on_cancelled: Callable[[], None]
```

In `PauseController.__init__`, replace `self._pending_reply = None` with:
```python
        # Deferred post-drain action for abort/wait pause OR release; held until
        # the scheduler drains. Single-consumer: only one may be armed at a time.
        self._pending_drain: _PendingDrain | None = None
        # True once GPU memory has actually been released (data plane). Distinct
        # from forward_blocked: PAUSED_ALL alone still permits DP idle forwards,
        # which touch weights and must be suppressed while released.
        self.released: bool = False
```

Add an `is_drain_pending` property and a generic `request_drain` used by both pause and release:
```python
    @property
    def is_drain_pending(self) -> bool:
        return self._pending_drain is not None

    def request_drain(
        self,
        *,
        abort_inflight: bool,
        on_drained: "Callable[[], None]",
        on_cancelled: "Callable[[], None]",
    ) -> bool:
        """Start a wait-style drain (PAUSED_NEW, cancel grammar-queued) and arm a
        post-drain action. Returns False if a drain is already pending (caller
        sends its own busy reply). ``abort_inflight=True`` also cancels in-flight
        requests (abort mode); False lets them finish (wait mode / release)."""
        if self._pending_drain is not None:
            return False
        self.state = PauseState.PAUSED_NEW
        self._pending_drain = _PendingDrain(on_drained, on_cancelled)
        self._cancel_grammar_pending = True
        if abort_inflight:
            self._abort_all_pending = True
        return True

    def set_released(self, released: bool) -> None:
        """Mark GPU memory released (freeze fully) or restored (unpause)."""
        self.released = released
        self.state = PauseState.PAUSED_ALL if released else PauseState.UNPAUSED
```

- [ ] **Step 3: Rewrite `handle_pause` to use `request_drain`**

Replace the abort/wait branch (lines ~165-172) so it composes `request_drain`; keep keep-mode and validation identical:
```python
    def handle_pause(self, req: PauseSchedulerReqInput) -> None:
        if req.mode not in ("abort", "wait", "keep"):
            self._send.send_pyobj(
                PauseSchedulerReqOutput(
                    success=False, message=f"invalid pause mode: {req.mode!r}"
                )
            )
            return

        if self._pending_drain is not None:
            self._send.send_pyobj(
                PauseSchedulerReqOutput(
                    success=False, message="a pause is already in progress"
                )
            )
            return

        if req.mode == "keep":
            self.state = PauseState.PAUSED_ALL
            self._send.send_pyobj(PauseSchedulerReqOutput(success=True))
            return

        self.request_drain(
            abort_inflight=(req.mode == "abort"),
            on_drained=lambda: self._send.send_pyobj(
                PauseSchedulerReqOutput(success=True)
            ),
            on_cancelled=lambda: self._send.send_pyobj(
                PauseSchedulerReqOutput(
                    success=False, message="resumed before pause drained"
                )
            ),
        )
```

- [ ] **Step 4: Update `handle_resume` and `maybe_finish_drain` to use `_pending_drain`**

```python
    def handle_resume(self, req: ResumeSchedulerReqInput) -> None:
        if self._pending_drain is not None:
            self._pending_drain.on_cancelled()
            self._pending_drain = None
        self.state = PauseState.UNPAUSED
        self.released = False
        self._abort_all_pending = False
        self._cancel_grammar_pending = False
        self._send.send_pyobj(ResumeSchedulerReqOutput(success=True))

    def maybe_finish_drain(self, scheduler) -> None:
        """Resolve a deferred pause/release action once the scheduler drains."""
        if self._pending_drain is None:
            return
        if not scheduler_drained(scheduler):
            return
        action = self._pending_drain
        self._pending_drain = None
        action.on_drained()
```

Note: `on_drained` is cleared *before* running so a release's `on_drained` can re-arm state (set `released`) without tripping the single-consumer guard.

- [ ] **Step 5: Run pause tests (must stay green)**

Run: `pytest test/runtime/test_pause_controller.py -v`
Expected: PASS — identical external behavior. If any test referenced `_pending_reply` directly, update it to `_pending_drain` (assert `is_drain_pending`).

- [ ] **Step 6: Commit**

```bash
git add python/tokenspeed/runtime/engine/pause.py test/runtime/test_pause_controller.py
git commit -s -m "refactor(pause): generalize deferred reply into _PendingDrain action + released flag"
```

---

## Phase 3 — MemoryOccupationController

### Task 4: MemoryOccupationController core (release/resume/is_sleeping)

**Files:**
- Create: `python/tokenspeed/runtime/engine/memory_occupation.py`
- Test: `test/runtime/test_memory_occupation_controller.py`

- [ ] **Step 1: Write the failing tests**

```python
# test/runtime/test_memory_occupation_controller.py
import pytest

from tokenspeed.runtime.engine.io_struct import (
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    IsSleepingReqInput,
    IsSleepingReqOutput,
)
from tokenspeed.runtime.engine.memory_occupation import MemoryOccupationController
from tokenspeed.runtime.engine.pause import PauseController


class FakeSender:
    def __init__(self):
        self.sent = []

    def send_pyobj(self, obj):
        self.sent.append(obj)


class FakeAdapter:
    def __init__(self):
        self.paused = []
        self.resumed = []

    def pause(self, tag=None):
        self.paused.append(tag)

    def resume(self, tag=None):
        self.resumed.append(tag)


class FakeScheduler:
    """Drains immediately (no in-flight)."""

    def waiting_size(self):
        return 0

    def decoding_size(self):
        return 0

    def prefilling_size(self):
        return 0

    def retract_count(self):
        return 0


def make(enable=True):
    send = FakeSender()
    pause = PauseController(send)
    adapter = FakeAdapter()
    calls = {"reset": 0, "kv_repair": 0}
    ctrl = MemoryOccupationController(
        send_func=send,
        pause_controller=pause,
        adapter=adapter,
        enabled=enable,
        reset_caches_fn=lambda: calls.__setitem__("reset", calls["reset"] + 1),
        kv_repair_fn=lambda: calls.__setitem__("kv_repair", calls["kv_repair"] + 1),
    )
    return ctrl, pause, adapter, send, calls


def test_release_defers_until_drain_then_frees_tags():
    ctrl, pause, adapter, send, calls = make()
    ctrl.handle_release(ReleaseMemoryOccupationReqInput(tags=None))
    # Deferred: nothing freed, no reply yet, but scheduler is pausing.
    assert adapter.paused == []
    assert send.sent == []
    assert pause.is_drain_pending
    # Event loop observes drain:
    pause.maybe_finish_drain(FakeScheduler())
    assert set(adapter.paused) == {"weights", "kv_cache"}
    assert calls["reset"] == 1  # prefix cache invalidated (KV discarded)
    assert pause.released is True
    assert isinstance(send.sent[-1], ReleaseMemoryOccupationReqOutput)
    assert send.sent[-1].success is True
    assert ctrl.is_sleeping is True


def test_resume_all_restores_and_unpauses():
    ctrl, pause, adapter, send, calls = make()
    ctrl.handle_release(ReleaseMemoryOccupationReqInput(tags=None))
    pause.maybe_finish_drain(FakeScheduler())
    send.sent.clear()
    ctrl.handle_resume(ResumeMemoryOccupationReqInput(tags=None))
    assert set(adapter.resumed) == {"weights", "kv_cache"}
    assert calls["kv_repair"] == 1
    assert pause.released is False
    assert pause.state.name == "UNPAUSED"
    assert send.sent[-1].success is True
    assert ctrl.is_sleeping is False


def test_multistage_resume_weights_then_kv():
    ctrl, pause, adapter, send, calls = make()
    ctrl.handle_release(ReleaseMemoryOccupationReqInput(tags=["weights", "kv_cache"]))
    pause.maybe_finish_drain(FakeScheduler())
    # Resume weights only: still sleeping, no KV repair yet.
    ctrl.handle_resume(ResumeMemoryOccupationReqInput(tags=["weights"]))
    assert adapter.resumed == ["weights"]
    assert calls["kv_repair"] == 0
    assert ctrl.is_sleeping is True
    assert pause.released is True
    # Resume kv_cache: KV repaired, fully awake.
    ctrl.handle_resume(ResumeMemoryOccupationReqInput(tags=["kv_cache"]))
    assert calls["kv_repair"] == 1
    assert ctrl.is_sleeping is False
    assert pause.state.name == "UNPAUSED"


def test_invalid_tag_rejected():
    ctrl, pause, adapter, send, calls = make()
    ctrl.handle_release(ReleaseMemoryOccupationReqInput(tags=["bogus"]))
    assert send.sent[-1].success is False
    assert not pause.is_drain_pending


def test_disabled_saver_rejects_release():
    ctrl, pause, adapter, send, calls = make(enable=False)
    ctrl.handle_release(ReleaseMemoryOccupationReqInput(tags=None))
    assert send.sent[-1].success is False
    assert "not enabled" in send.sent[-1].message
    assert adapter.paused == []


def test_double_release_tag_rejected():
    ctrl, pause, adapter, send, calls = make()
    ctrl.handle_release(ReleaseMemoryOccupationReqInput(tags=["weights"]))
    pause.maybe_finish_drain(FakeScheduler())
    ctrl.handle_release(ReleaseMemoryOccupationReqInput(tags=["weights"]))
    assert send.sent[-1].success is False  # already released


def test_resume_not_released_tag_rejected():
    ctrl, pause, adapter, send, calls = make()
    ctrl.handle_resume(ResumeMemoryOccupationReqInput(tags=["weights"]))
    assert send.sent[-1].success is False  # nothing released


def test_resume_before_release_drained_cancels():
    ctrl, pause, adapter, send, calls = make()
    # Scheduler that never drains:
    class Busy(FakeScheduler):
        def decoding_size(self):
            return 1

    ctrl.handle_release(ReleaseMemoryOccupationReqInput(tags=None))
    pause.maybe_finish_drain(Busy())  # not drained → no release
    assert adapter.paused == []
    # A scheduler-resume arrives first → release is cancelled with failure.
    from tokenspeed.runtime.engine.io_struct import ResumeSchedulerReqInput
    pause.handle_resume(ResumeSchedulerReqInput())
    fails = [s for s in send.sent if isinstance(s, ReleaseMemoryOccupationReqOutput)]
    assert fails and fails[-1].success is False


def test_is_sleeping_handler():
    ctrl, pause, adapter, send, calls = make()
    ctrl.handle_is_sleeping(IsSleepingReqInput())
    assert isinstance(send.sent[-1], IsSleepingReqOutput)
    assert send.sent[-1].is_sleeping is False
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest test/runtime/test_memory_occupation_controller.py -v`
Expected: FAIL — `ModuleNotFoundError: memory_occupation`.

- [ ] **Step 3: Implement `memory_occupation.py`**

```python
# python/tokenspeed/runtime/engine/memory_occupation.py
# SPDX-License-Identifier: Apache-2.0
"""GPU-memory data plane for sleep/wake (release/resume_memory_occupation).

Pairs with the control-plane :class:`PauseController`. A *release* pauses and
drains the scheduler (delegated to the pause controller) and only then frees GPU
memory via the torch_memory_saver adapter; a *resume* re-maps memory and, when
the KV region returns, repairs the KV cache. Tags (`weights`, `kv_cache`) are
freed/restored independently so the RL loop can resume weights, push fresh
weights, then resume the KV cache.
"""

from __future__ import annotations

from collections.abc import Callable

from tokenspeed.runtime.engine.io_struct import (
    IsSleepingReqInput,
    IsSleepingReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
)
from tokenspeed.runtime.engine.pause import PauseController

VALID_TAGS = ("weights", "kv_cache")


def _normalize_tags(tags: list[str] | None) -> tuple[list[str] | None, str | None]:
    """Return (ordered_tags, error). None ⇒ all tags. Order: weights before
    kv_cache on release (free big weights last is fine; order is for determinism)."""
    if tags is None:
        return list(VALID_TAGS), None
    bad = [t for t in tags if t not in VALID_TAGS]
    if bad:
        return None, f"invalid tags: {bad!r}; valid: {list(VALID_TAGS)}"
    # Deterministic order.
    return [t for t in VALID_TAGS if t in tags], None


class MemoryOccupationController:
    """Owns the data-plane release/resume for one scheduler event loop."""

    def __init__(
        self,
        *,
        send_func,
        pause_controller: PauseController,
        adapter,
        enabled: bool,
        reset_caches_fn: Callable[[], None],
        kv_repair_fn: Callable[[], None],
    ) -> None:
        self._send = send_func
        self._pause = pause_controller
        self._adapter = adapter
        self._enabled = enabled
        self._reset_caches = reset_caches_fn
        self._kv_repair = kv_repair_fn
        self.released_tags: set[str] = set()

    @property
    def is_sleeping(self) -> bool:
        return bool(self.released_tags)

    # -- control-request handlers (driven by the request handler) -------------

    def handle_release(self, req: ReleaseMemoryOccupationReqInput) -> None:
        if not self._enabled:
            self._fail_release("memory saver not enabled (--enable-memory-saver)")
            return
        tags, err = _normalize_tags(req.tags)
        if err is not None:
            self._fail_release(err)
            return
        already = [t for t in tags if t in self.released_tags]
        if already:
            self._fail_release(f"tags already released: {already!r}")
            return
        if self._pause.is_drain_pending:
            self._fail_release("a pause or release is already in progress")
            return
        # Defer the actual free until the scheduler drains.
        self._pause.request_drain(
            abort_inflight=False,  # wait: let in-flight finish (RL rollout is idle)
            on_drained=lambda: self._finish_release(tags),
            on_cancelled=lambda: self._fail_release("resumed before release drained"),
        )

    def _finish_release(self, tags: list[str]) -> None:
        if "kv_cache" in tags:
            # KV is discarded; any prefix-cache entry pointing at it is stale.
            self._reset_caches()
        for tag in tags:
            self._adapter.pause(tag=tag)
            self.released_tags.add(tag)
        self._pause.set_released(True)
        self._send.send_pyobj(ReleaseMemoryOccupationReqOutput(success=True))

    def _fail_release(self, message: str) -> None:
        self._send.send_pyobj(
            ReleaseMemoryOccupationReqOutput(success=False, message=message)
        )

    def handle_resume(self, req: ResumeMemoryOccupationReqInput) -> None:
        tags, err = _normalize_tags(req.tags)
        if err is not None:
            self._send.send_pyobj(
                ResumeMemoryOccupationReqOutput(success=False, message=err)
            )
            return
        not_released = [t for t in tags if t not in self.released_tags]
        if not_released:
            self._send.send_pyobj(
                ResumeMemoryOccupationReqOutput(
                    success=False, message=f"tags not released: {not_released!r}"
                )
            )
            return
        for tag in tags:
            self._adapter.resume(tag=tag)
            self.released_tags.discard(tag)
        if "kv_cache" in tags:
            self._kv_repair()
        if not self.released_tags:
            self._pause.set_released(False)  # fully awake → unpause
        self._send.send_pyobj(ResumeMemoryOccupationReqOutput(success=True))

    def handle_is_sleeping(self, req: IsSleepingReqInput) -> None:
        self._send.send_pyobj(IsSleepingReqOutput(is_sleeping=self.is_sleeping))
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest test/runtime/test_memory_occupation_controller.py -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add python/tokenspeed/runtime/engine/memory_occupation.py test/runtime/test_memory_occupation_controller.py
git commit -s -m "feat(engine): MemoryOccupationController for release/resume_memory_occupation"
```

---

## Phase 4 — Scheduler-process wiring

### Task 5: request_handler dispatch

**Files:**
- Modify: `python/tokenspeed/runtime/engine/request_handler.py` (imports ~38, ctor, dispatch ~186)

- [ ] **Step 1: Add imports**

In the io_struct import block (around line 38), add:
```python
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    IsSleepingReqInput,
```

- [ ] **Step 2: Accept the controller in the constructor**

Add a `memory_controller` parameter to `RequestHandler.__init__` (mirroring `pause_controller`) and store `self.memory_controller = memory_controller`.

- [ ] **Step 3: Add dispatch branches**

After the `IsSchedulerPausedReqInput` branch (line 186), add:
```python
            elif isinstance(recv_req, ReleaseMemoryOccupationReqInput):
                # Deferred: pauses + drains, then frees GPU memory and replies.
                self.memory_controller.handle_release(recv_req)
            elif isinstance(recv_req, ResumeMemoryOccupationReqInput):
                self.memory_controller.handle_resume(recv_req)
            elif isinstance(recv_req, IsSleepingReqInput):
                self.memory_controller.handle_is_sleeping(recv_req)
```

- [ ] **Step 4: Run the handler/pause tests**

Run: `pytest test/runtime/test_pause_controller.py test/runtime/test_memory_occupation_controller.py -v`
Expected: PASS (no regressions; new branches are covered by controller tests).

- [ ] **Step 5: Commit**

```bash
git add python/tokenspeed/runtime/engine/request_handler.py
git commit -s -m "feat(request-handler): dispatch release/resume/is_sleeping to MemoryOccupationController"
```

### Task 6: event_loop wiring + DP idle-forward gate

**Files:**
- Modify: `python/tokenspeed/runtime/engine/event_loop.py` (ctor ~372-400, `_paused_idle_step` ~1182-1189)

- [ ] **Step 1: Construct the controller and pass it to the handler**

After `self._pause = PauseController(self.send_to_tokenizer)` (line 372), add:
```python
        from tokenspeed.runtime.engine.memory_occupation import (
            MemoryOccupationController,
        )
        from tokenspeed.runtime.utils.torch_memory_saver_adapter import (
            TorchMemorySaverAdapter,
        )

        self._memory = MemoryOccupationController(
            send_func=self.send_to_tokenizer,
            pause_controller=self._pause,
            adapter=TorchMemorySaverAdapter.create(
                enable=self.server_args.enable_memory_saver
            ),
            enabled=self.server_args.enable_memory_saver,
            reset_caches_fn=self._reset_caches_for_release,
            kv_repair_fn=self._kv_repair_after_wake,
        )
```
Then add `memory_controller=self._memory,` to the `RequestHandler(...)` call (line 387-400).

- [ ] **Step 2: Add the two callbacks**

Add methods near `_paused_idle_step`. Wire `_reset_caches_for_release` to the same prefix-cache invalidation the FlushCache path uses (locate it first: `grep -rn "reset_prefix\|radix\|flush" python/tokenspeed/runtime/engine/event_loop.py` and the C++ binding). If no Python-reachable reset exists yet, make it a best-effort no-op **and log a warning**, and rely on the GPU integration test (Task 13) to confirm whether stale prefix hits actually occur in the RL flow:
```python
    def _reset_caches_for_release(self) -> None:
        """Invalidate prefix/radix cache before KV is discarded. KV pages are
        remapped+zeroed on wake, so any retained prefix entry would be stale."""
        reset = getattr(self.scheduler, "reset_prefix_cache", None)
        if callable(reset):
            reset()
        else:
            logger.warning(
                "release_memory_occupation: no scheduler prefix-cache reset "
                "available; relying on caller to not depend on pre-sleep cache."
            )

    def _kv_repair_after_wake(self) -> None:
        """Zero remapped KV buffers (garbage after re-map). FP8 KV scales ride
        with the weights region, so no scale reset is needed here."""
        pool = getattr(self.model_executor, "token_to_kv_pool", None)
        if pool is None:
            return
        for buf in list(getattr(pool, "k_buffer", [])) + list(
            getattr(pool, "v_buffer", [])
        ):
            buf.zero_()
```
> **Verify on nv2 (Task 0/13):** the exact KV-pool attribute (`self.model_executor.token_to_kv_pool` vs `self.model_executor.model_runner...`) and buffer names (`k_buffer/v_buffer` for MHA; MLA uses a single `kv_buffer`). Extend the zero loop to cover MLA's buffer name.

- [ ] **Step 3: Gate DP idle forward while released**

In `_paused_idle_step` (line 1182-1189), change the idle-forward condition:
```python
        if self.has_dp:
            dp_metadata = self._dp_sync_and_check(None)
            if dp_metadata.need_idle_forward and not self._pause.released:
                self.model_executor.execute_idle_forward(
                    dp_metadata.global_num_tokens,
                    dp_metadata.global_batch_size,
                    dp_metadata.all_decode_or_idle,
                )
```
Rationale: an idle forward runs the model and touches weights; while `released`, weights may be unmapped. All DP ranks release together, so skipping idle forward stays consistent across ranks.

- [ ] **Step 4: Smoke-import the event loop module**

Run: `python -c "import tokenspeed.runtime.engine.event_loop"`
Expected: no ImportError. (Full event-loop behavior is covered by the GPU integration test; unit-level it has heavy deps.)

- [ ] **Step 5: Commit**

```bash
git add python/tokenspeed/runtime/engine/event_loop.py
git commit -s -m "feat(event-loop): wire MemoryOccupationController; skip DP idle forward while released"
```

---

## Phase 5 — Client / engine / HTTP surface

### Task 7: client returns outputs + is_sleeping

**Files:**
- Modify: `python/tokenspeed/runtime/engine/scheduler_control_client.py` (init ~176, dispatcher ~240, methods ~407-419)
- Modify: `python/tokenspeed/runtime/engine/protocol.py` (~189)
- Modify: `python/tokenspeed/runtime/engine/io_struct.py` import site if needed

- [ ] **Step 1: Add the is_sleeping communicator + dispatcher entry**

In `init_communicators`, after `is_scheduler_paused_communicator` (line 178):
```python
        self.is_sleeping_communicator = _Communicator(
            self.engine_core_client.send_to_scheduler, server_args.mapping.attn.dp_size
        )
```
In `_get_communicator_dispatcher`, add (import `IsSleepingReqOutput` at top):
```python
                (
                    IsSleepingReqOutput,
                    self.is_sleeping_communicator.handle_recv,
                ),
```

- [ ] **Step 2: Make release/resume return the output; add is_sleeping**

Replace the two methods (lines 407-419) and add a third:
```python
    async def release_memory_occupation(
        self: AsyncLLM,
        obj: ReleaseMemoryOccupationReqInput,
    ) -> ReleaseMemoryOccupationReqOutput:
        self.auto_create_handle_loop()
        return (await self.release_memory_occupation_communicator(obj))[0]

    async def resume_memory_occupation(
        self: AsyncLLM,
        obj: ResumeMemoryOccupationReqInput,
    ) -> ResumeMemoryOccupationReqOutput:
        self.auto_create_handle_loop()
        return (await self.resume_memory_occupation_communicator(obj))[0]

    async def is_sleeping(self: AsyncLLM) -> bool:
        self.auto_create_handle_loop()
        result = (await self.is_sleeping_communicator(IsSleepingReqInput()))[0]
        return result.is_sleeping
```
Add `IsSleepingReqInput`, `IsSleepingReqOutput`, `ReleaseMemoryOccupationReqOutput`, `ResumeMemoryOccupationReqOutput` to the io_struct imports.

- [ ] **Step 3: protocol stub**

In `protocol.py` after the memory occupation block (~194), add:
```python
    async def is_sleeping(self) -> bool: ...
```

- [ ] **Step 4: Import-smoke**

Run: `python -c "import tokenspeed.runtime.engine.scheduler_control_client"`
Expected: no ImportError.

- [ ] **Step 5: Commit**

```bash
git add python/tokenspeed/runtime/engine/scheduler_control_client.py python/tokenspeed/runtime/engine/protocol.py
git commit -s -m "feat(client): return release/resume outputs; add is_sleeping RPC"
```

### Task 8: engine + engine_base surface

**Files:**
- Modify: `python/tokenspeed/runtime/entrypoints/engine.py` (~418-424)
- Modify: `python/tokenspeed/runtime/entrypoints/engine_base.py` (~74-80)

- [ ] **Step 1: engine.py — add is_sleeping(); release/resume already pass tags**

After `resume_memory_occupation` (line 424), add:
```python
    def is_sleeping(self) -> bool:
        """Return whether GPU memory is currently released (data-plane sleep)."""
        return self.llm.run(self.tokenizer_manager.is_sleeping())
```

- [ ] **Step 2: engine_base.py — widen abstract signatures**

Replace lines 74-80:
```python
    @abstractmethod
    def release_memory_occupation(self, tags: list[str] | None = None) -> None:
        """Release GPU memory occupation temporarily (optionally by tag)."""

    @abstractmethod
    def resume_memory_occupation(self, tags: list[str] | None = None) -> None:
        """Resume GPU memory occupation previously released (optionally by tag)."""

    @abstractmethod
    def is_sleeping(self) -> bool:
        """Return whether GPU memory is currently released."""
```

- [ ] **Step 3: Import-smoke**

Run: `python -c "import tokenspeed.runtime.entrypoints.engine"`
Expected: no ImportError. (If `engine.py` subclasses `EngineBase`, ensure all three abstract methods are implemented — they are.)

- [ ] **Step 4: Commit**

```bash
git add python/tokenspeed/runtime/entrypoints/engine.py python/tokenspeed/runtime/entrypoints/engine_base.py
git commit -s -m "feat(engine): is_sleeping(); widen release/resume signatures for tags"
```

### Task 9: HTTP routes

**Files:**
- Modify: `python/tokenspeed/runtime/entrypoints/http_server.py`
- Test: `test/runtime/test_http_sleep_routes.py` (create, FastAPI TestClient)

- [ ] **Step 1: Write the failing test** (mock engine client; assert parse+forward)

```python
# test/runtime/test_http_sleep_routes.py
# NOTE: adapt the app factory import to match http_server.py's actual builder.
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from tokenspeed.runtime.entrypoints import http_server


class FakeEngineClient:
    def __init__(self):
        self.calls = []

    async def release_memory_occupation(self, obj):
        self.calls.append(("release", obj.tags))

    async def resume_memory_occupation(self, obj):
        self.calls.append(("resume", obj.tags))

    async def is_sleeping(self):
        self.calls.append(("is_sleeping", None))
        return True


def make_client():
    app = http_server.build_app_for_test()  # add this thin factory in Step 3
    app.state.engine_client = FakeEngineClient()
    return TestClient(app), app.state.engine_client


def test_release_parses_tags():
    client, eng = make_client()
    r = client.post("/release_memory_occupation?tags=weights&tags=kv_cache")
    assert r.status_code == 200
    assert eng.calls[-1] == ("release", ["weights", "kv_cache"])


def test_release_no_tags_is_none():
    client, eng = make_client()
    client.post("/release_memory_occupation")
    assert eng.calls[-1] == ("release", None)


def test_is_sleeping_returns_json():
    client, eng = make_client()
    r = client.get("/is_sleeping")
    assert r.json() == {"is_sleeping": True}
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest test/runtime/test_http_sleep_routes.py -v`
Expected: FAIL — routes / `build_app_for_test` missing.

- [ ] **Step 3: Add the routes**

In `http_server.py`, following the existing direct-engine-call pattern (like `/abort`), add (using the module's existing app/router object and `engine_client` accessor):
```python
@app.post("/release_memory_occupation")
async def release_memory_occupation(raw_request: Request):
    tags = raw_request.query_params.getlist("tags") or None
    obj = ReleaseMemoryOccupationReqInput(tags=tags)
    out = await raw_request.app.state.engine_client.release_memory_occupation(obj)
    return ORJSONResponse(_as_dict(out))


@app.post("/resume_memory_occupation")
async def resume_memory_occupation(raw_request: Request):
    tags = raw_request.query_params.getlist("tags") or None
    obj = ResumeMemoryOccupationReqInput(tags=tags)
    out = await raw_request.app.state.engine_client.resume_memory_occupation(obj)
    return ORJSONResponse(_as_dict(out))


@app.get("/is_sleeping")
async def is_sleeping(raw_request: Request):
    val = await raw_request.app.state.engine_client.is_sleeping()
    return ORJSONResponse({"is_sleeping": val})
```
Match the file's actual response helper (`ORJSONResponse`/`JSONResponse`) and engine accessor. Add a minimal `build_app_for_test()` factory if the app isn't otherwise constructible without a live engine, and a `_as_dict(out)` that tolerates `None` (older clients returned None). Import `ReleaseMemoryOccupationReqInput`/`ResumeMemoryOccupationReqInput`.
> **Dev-mode gate:** check whether http_server gates dev/control routes behind an env flag (vLLM uses `VLLM_SERVER_DEV_MODE`). If TokenSpeed has an equivalent, register these routes behind it.

- [ ] **Step 4: Run to verify pass**

Run: `pytest test/runtime/test_http_sleep_routes.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/tokenspeed/runtime/entrypoints/http_server.py test/runtime/test_http_sleep_routes.py
git commit -s -m "feat(http): /release_memory_occupation /resume_memory_occupation /is_sleeping"
```

---

## Phase 6 — Tag the allocation regions

### Task 10: Tag weights and KV regions

**Files:**
- Modify: `python/tokenspeed/runtime/execution/weight_loader.py:86`
- Modify: `python/tokenspeed/runtime/layers/attention/kv_cache/mha.py:102`
- Modify: `python/tokenspeed/runtime/layers/attention/kv_cache/mla.py:85`
- Modify: `python/tokenspeed/runtime/cache/req_to_token_pool.py:65`

- [ ] **Step 1: Tag weights (with CPU backup so weights restore on wake)**

`weight_loader.py:86`: `with memory_saver_adapter.region():` → `with memory_saver_adapter.region(tag="weights", enable_cpu_backup=True):`

- [ ] **Step 2: Tag KV (MHA + MLA + req_to_token) — discard on sleep (no CPU backup)**

- `mha.py:102`: `with self.memory_saver_adapter.region():` → `region(tag="kv_cache", enable_cpu_backup=False)`
- `mla.py:85`: `with memory_saver_adapter.region():` → `region(tag="kv_cache", enable_cpu_backup=False)`
- `req_to_token_pool.py:65`: `with memory_saver_adapter.region():` → `region(tag="kv_cache", enable_cpu_backup=False)`

- [ ] **Step 3: Import-smoke each module**

Run: `python -c "import tokenspeed.runtime.execution.weight_loader, tokenspeed.runtime.layers.attention.kv_cache.mha, tokenspeed.runtime.layers.attention.kv_cache.mla, tokenspeed.runtime.cache.req_to_token_pool"`
Expected: no ImportError.

- [ ] **Step 4: Commit**

```bash
git add python/tokenspeed/runtime/execution/weight_loader.py python/tokenspeed/runtime/layers/attention/kv_cache/mha.py python/tokenspeed/runtime/layers/attention/kv_cache/mla.py python/tokenspeed/runtime/cache/req_to_token_pool.py
git commit -s -m "feat(memory-saver): tag weights vs kv_cache allocation regions"
```

### Task 11: Tag deepseek_v4 KV region (nv2-informed)

**Files:**
- Modify: `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py` (~767-791)

- [ ] **Step 1: Wrap the KV allocation**

Using the allocation site identified in Task 0 Step 4, stop discarding `enable_memory_saver` and wrap the KV buffer allocation:
```python
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        with self.memory_saver_adapter.region(tag="kv_cache"):
            # ... existing KV buffer allocation ...
```
(Import `TorchMemorySaverAdapter` if not present.)

- [ ] **Step 2: Import-smoke**

Run: `python -c "import tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4"`
Expected: no ImportError.

- [ ] **Step 3: Commit**

```bash
git add python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py
git commit -s -m "feat(memory-saver): tag deepseek_v4 KV region for sleep/wake"
```

### Task 12: Pin torch_memory_saver + ensure it's in the runner image

**Files:**
- Modify: `python/pyproject.toml`
- Modify: the runner image build (`docker/`) OR document a runtime install

- [ ] **Step 1: Add a pinned optional dependency**

Add `torch_memory_saver==0.0.9.post1` (confirmed tag-capable + cu13 binaries bundled, Task 0) to an optional/extras group in `python/pyproject.toml` (keep it optional — it's CUDA-only and the adapter import-guards it). If a group like `[project.optional-dependencies].gpu` exists, add it there.

- [ ] **Step 2: Ensure the runner image installs it**

`torch_memory_saver` is NOT in `lightseekorg/tokenspeed-runner:latest` (Task 0). Add `pip install torch_memory_saver==0.0.9.post1` to the image's Dockerfile under `docker/`. For interim nv2 validation it can be installed at container start with `--break-system-packages` (see Task 14 Step 0).

- [ ] **Step 3: Commit**

```bash
git add python/pyproject.toml docker/
git commit -s -m "build: add tag-capable torch_memory_saver==0.0.9.post1 (optional + runner image)"
```

---

## Phase 7 — Run the unit suite + pre-commit

### Task 13: Full local check

- [ ] **Step 1: Run all new + touched unit tests**

Run:
```bash
pytest test/runtime/test_io_struct_memory_occupation.py \
       test/runtime/test_memory_saver_adapter.py \
       test/runtime/test_pause_controller.py \
       test/runtime/test_memory_occupation_controller.py \
       test/runtime/test_http_sleep_routes.py -v
```
Expected: all PASS.

- [ ] **Step 2: pre-commit**

Run: `pre-commit run --all-files`
Expected: clean (fix formatting if flagged), then re-run.

- [ ] **Step 3: Commit any formatting**

```bash
git add -A && git commit -s -m "style: pre-commit formatting for sleep/wake"
```

---

## Phase 8 — GPU integration on nv2 (multiple cases)

### Task 14: GPU integration tests on B200

**Files:**
- Create: `test/runtime/test_sleep_wakeup_gpu.py` (GPU-gated, e.g. `@pytest.mark.skipif(not torch.cuda.is_available())` plus an env opt-in like the existing GPU tests).

Deploy per memory `nv2-logprobs-validation-env` / `pause-api-and-nv2-pyvalidation`: this is **pure Python**, so shadow the package over the existing install and reuse the prebuilt scheduler `.so` — no kernel/scheduler rebuild. Use host `v2` (b300), image `lightseekorg/tokenspeed-runner:latest`, `docker run --gpus all --ipc=host`.

- [ ] **Step 0: Install torch_memory_saver in the container**

`pip install --break-system-packages torch_memory_saver==0.0.9.post1` (not preinstalled in the image). Confirm `python3 -c "from torch_memory_saver.utils import _detect_cuda_major; print(_detect_cuda_major())"` → 13. The scheduler subprocess gets `LD_PRELOAD` automatically via the existing `configure_subprocess()` wrap (engine.py:528).

- [ ] **Step 1: Launch a small model with sleep enabled**

Start the engine with `--enable-memory-saver` on a small model (e.g. Qwen2-1.5B) on 1 GPU. Confirm `is_sleeping()` is False at boot.

- [ ] **Step 2: Case A — full release/resume frees and restores memory**

```python
free0 = torch.cuda.mem_get_info()[0]
engine.release_memory_occupation()            # tags=None → weights+kv
free1 = torch.cuda.mem_get_info()[0]
assert free1 > free0                           # GPU memory freed
assert engine.is_sleeping() is True
engine.resume_memory_occupation()
assert engine.is_sleeping() is False
# generation after wake is coherent:
out = engine.generate("The capital of France is", max_new_tokens=8)
assert "Paris" in out
```

- [ ] **Step 3: Case B — generation works identically before vs after a sleep cycle**

Generate a fixed prompt with greedy decoding before sleep and after wake; assert token-identical output (weights restored byte-for-byte from CPU).

- [ ] **Step 4: Case C — RL multi-stage tag flow**

```python
engine.release_memory_occupation(tags=["weights", "kv_cache"])
assert engine.is_sleeping()
engine.resume_memory_occupation(tags=["weights"])
assert engine.is_sleeping()                    # kv still released
engine.update_weights_from_tensor(new_named_tensors)
engine.resume_memory_occupation(tags=["kv_cache"])
assert not engine.is_sleeping()
out = engine.generate(prompt, max_new_tokens=8) # reflects updated weights
```

- [ ] **Step 5: Case D — error paths**

Assert: resuming a not-released tag → `success=False`; double-release a tag → `success=False`; release with `--enable-memory-saver` off → `success=False, "not enabled"`.

- [ ] **Step 6: Case E — FP8 KV cache**

Launch with `--kv-cache-dtype fp8_e4m3`; run Case A; assert post-wake generation is coherent (validates KV repair + scales riding with weights).

- [ ] **Step 7: Case F — CUDA graphs survive remap**

Launch with cudagraphs enabled (default); run Case B; assert token-identical output (validates same-vaddr remap keeps captured graphs valid).

- [ ] **Step 8: Case G — TP and DP**

Run Case A under `--tp 2` and under data parallelism; assert no NCCL hang while `released` (validates the DP idle-forward gate) and coherent output after wake.

- [ ] **Step 9: Record results + commit the test**

```bash
git add test/runtime/test_sleep_wakeup_gpu.py
git commit -s -m "test(gpu): sleep/wake integration cases (release/resume, tags, RL, fp8, cudagraph, TP/DP)"
```

- [ ] **Step 10: Open PR**

Push the branch and open a PR summarizing the design doc, the nv2 findings (Task 0), and the GPU case results.

---

## Self-review notes (gaps flagged for execution)

- **Prefix-cache reset on release** (`_reset_caches_for_release`): FlushCache is currently an ack-only stub, so a Python-reachable scheduler reset may not exist. Task 6 makes it best-effort + warning; Task 14 Case B/C confirm whether stale prefix hits occur and, if so, a follow-up wires a real reset.
- **KV-pool attribute & MLA buffer name**: Task 6's `_kv_repair_after_wake` uses `token_to_kv_pool.k_buffer/v_buffer`; verify the exact attribute path and add MLA's `kv_buffer` on nv2 (Task 0/14).
- **torch_memory_saver tag API**: Task 0 gates Task 2/12; if `tag=` isn't supported, use the per-tag-instance fallback in Task 2.
- **HTTP app factory**: Task 9 assumes a constructible app; adapt `build_app_for_test`/route registration to http_server.py's actual structure.
