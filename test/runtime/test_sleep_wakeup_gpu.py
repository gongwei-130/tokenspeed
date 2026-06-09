# SPDX-License-Identifier: Apache-2.0
"""GPU integration cases for the Sleep/Wake Up API (release/resume_memory_occupation).

Run on a CUDA box with torch_memory_saver installed and a small model. Not part
of the default unit suite (needs a GPU + model download). Driven as a script:

    CUDA_VISIBLE_DEVICES=2 python3 test/runtime/test_sleep_wakeup_gpu.py [MODEL]

Cases: A full release/resume frees+restores GPU memory; B generation is
token-identical across a sleep cycle (weights restored byte-exact); C RL
multi-stage tag flow (release -> resume weights -> resume kv_cache); D error
paths (resume not-released, double release).
"""

import os
import subprocess
import sys

# The engine is pinned to a physical GPU via CUDA_VISIBLE_DEVICES, but nvidia-smi
# ignores that var and indexes physical GPUs — so measure the physical index the
# engine actually uses, not 0.
_PHYS_GPU = int((os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0])


def gpu_used_mib(index: int = _PHYS_GPU) -> int:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            str(index),
        ]
    )
    return int(out.decode().strip().splitlines()[0])


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2-0.5B-Instruct"
    from tokenspeed.runtime.entrypoints.engine import Engine

    # CUDA_VISIBLE_DEVICES pins us to one physical GPU; index 0 here.
    engine = Engine(
        model=model,
        enable_memory_saver=True,
        gpu_memory_utilization=float(os.environ.get("GMU", "0.1")),
        max_model_len=2048,
        trust_remote_code=True,
        log_level="info",
    )
    prompt = "The capital of France is"
    sp = {"temperature": 0.0, "max_new_tokens": 16}

    print("[boot] is_sleeping:", engine.is_sleeping())
    base = engine.generate(prompt, sp)
    base_text = base["text"] if isinstance(base, dict) else base[0]["text"]
    print("[baseline]", repr(base_text))

    used0 = gpu_used_mib()
    print(f"[memA] used before release: {used0} MiB")

    # --- Case A: full release frees GPU memory ---
    r = engine.release_memory_occupation()
    print("[A] release ->", r, "is_sleeping:", engine.is_sleeping())
    used1 = gpu_used_mib()
    print(f"[memA] used after release: {used1} MiB (freed {used0 - used1} MiB)")
    assert engine.is_sleeping() is True, "should be sleeping after release"
    assert used1 < used0, "release must free GPU memory"

    engine.resume_memory_occupation()
    print("[A] resume -> is_sleeping:", engine.is_sleeping())
    used2 = gpu_used_mib()
    print(f"[memA] used after resume: {used2} MiB")
    assert engine.is_sleeping() is False, "should be awake after resume"

    # --- Case B: token-identical generation across a sleep cycle ---
    after = engine.generate(prompt, sp)
    after_text = after["text"] if isinstance(after, dict) else after[0]["text"]
    print("[B] after-wake:", repr(after_text))
    assert after_text == base_text, f"output changed across sleep: {base_text!r} != {after_text!r}"
    print("[B] token-identical across sleep cycle: OK")

    # --- Case C: RL multi-stage tag flow ---
    engine.release_memory_occupation(tags=["weights", "kv_cache"])
    assert engine.is_sleeping() is True
    engine.resume_memory_occupation(tags=["weights"])
    print("[C] after resume weights -> is_sleeping:", engine.is_sleeping())
    assert engine.is_sleeping() is True, "kv still released => still sleeping"
    engine.resume_memory_occupation(tags=["kv_cache"])
    assert engine.is_sleeping() is False, "fully awake after kv resume"
    c_text = engine.generate(prompt, sp)
    c_text = c_text["text"] if isinstance(c_text, dict) else c_text[0]["text"]
    print("[C] multi-stage wake generate:", repr(c_text))
    print("[C] multi-stage tag flow: OK")

    # --- Case D: error paths ---
    d1 = engine.resume_memory_occupation(tags=["weights"])  # nothing released
    print("[D] resume not-released ->", d1)
    assert getattr(d1, "success", True) is False, "resume of not-released tag must fail"
    engine.release_memory_occupation(tags=["weights"])
    d2 = engine.release_memory_occupation(tags=["weights"])  # double release
    print("[D] double release ->", d2)
    assert getattr(d2, "success", True) is False, "double release must fail"
    engine.resume_memory_occupation(tags=["weights"])  # clean up
    print("[D] error paths: OK")

    print("\nALL GPU CASES PASSED")
    engine.shutdown()


if __name__ == "__main__":
    main()
