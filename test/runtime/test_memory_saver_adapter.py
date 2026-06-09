# SPDX-License-Identifier: Apache-2.0
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


def test_noop_adapter_accepts_tag_kwarg():
    a = TorchMemorySaverAdapter.create(enable=False)
    with a.region(tag="weights", enable_cpu_backup=True):
        pass
    with a.region(tag="kv_cache", enable_cpu_backup=False):
        pass
    a.pause(tag="weights")  # no-op, must not raise
    a.resume(tag="weights")  # no-op, must not raise


def test_noop_adapter_region_without_args():
    a = TorchMemorySaverAdapter.create(enable=False)
    with a.region():
        pass
