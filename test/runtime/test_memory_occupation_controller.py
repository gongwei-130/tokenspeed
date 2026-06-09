# SPDX-License-Identifier: Apache-2.0
from tokenspeed.runtime.engine.io_struct import (
    IsSleepingReqInput,
    IsSleepingReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    ResumeSchedulerReqInput,
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


class DrainedScheduler:
    """No in-flight work — drains immediately."""

    def waiting_size(self):
        return 0

    def decoding_size(self):
        return 0

    def prefilling_size(self):
        return 0

    def retract_count(self):
        return 0


class BusyScheduler(DrainedScheduler):
    def decoding_size(self):
        return 1


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
    # Deferred: nothing freed, no reply yet, scheduler pausing.
    assert adapter.paused == []
    assert send.sent == []
    assert pause.is_drain_pending
    # Event loop observes the drain:
    pause.maybe_finish_drain(DrainedScheduler())
    assert set(adapter.paused) == {"weights", "kv_cache"}
    assert calls["reset"] == 1  # prefix cache invalidated (KV discarded)
    assert pause.released is True
    assert isinstance(send.sent[-1], ReleaseMemoryOccupationReqOutput)
    assert send.sent[-1].success is True
    assert ctrl.is_sleeping is True


def test_resume_all_restores_and_unpauses():
    ctrl, pause, adapter, send, calls = make()
    ctrl.handle_release(ReleaseMemoryOccupationReqInput(tags=None))
    pause.maybe_finish_drain(DrainedScheduler())
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
    pause.maybe_finish_drain(DrainedScheduler())
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
    assert adapter.paused == []


def test_disabled_saver_rejects_release():
    ctrl, pause, adapter, send, calls = make(enable=False)
    ctrl.handle_release(ReleaseMemoryOccupationReqInput(tags=None))
    assert send.sent[-1].success is False
    assert "not enabled" in send.sent[-1].message
    assert adapter.paused == []
    assert not pause.is_drain_pending


def test_double_release_tag_rejected():
    ctrl, pause, adapter, send, calls = make()
    ctrl.handle_release(ReleaseMemoryOccupationReqInput(tags=["weights"]))
    pause.maybe_finish_drain(DrainedScheduler())
    ctrl.handle_release(ReleaseMemoryOccupationReqInput(tags=["weights"]))
    assert send.sent[-1].success is False  # already released
    assert adapter.paused == ["weights"]  # not paused twice


def test_resume_not_released_tag_rejected():
    ctrl, pause, adapter, send, calls = make()
    ctrl.handle_resume(ResumeMemoryOccupationReqInput(tags=["weights"]))
    assert send.sent[-1].success is False  # nothing released
    assert adapter.resumed == []


def test_resume_before_release_drained_cancels():
    ctrl, pause, adapter, send, calls = make()
    ctrl.handle_release(ReleaseMemoryOccupationReqInput(tags=None))
    pause.maybe_finish_drain(BusyScheduler())  # not drained → no release
    assert adapter.paused == []
    # A scheduler-resume arrives first → release is cancelled with failure.
    pause.handle_resume(ResumeSchedulerReqInput())
    fails = [s for s in send.sent if isinstance(s, ReleaseMemoryOccupationReqOutput)]
    assert fails and fails[-1].success is False
    assert pause.released is False


def test_release_while_drain_pending_rejected():
    ctrl, pause, adapter, send, calls = make()
    ctrl.handle_release(ReleaseMemoryOccupationReqInput(tags=["weights"]))
    # Second release before the first drains:
    ctrl.handle_release(ReleaseMemoryOccupationReqInput(tags=["kv_cache"]))
    assert send.sent[-1].success is False
    assert "in progress" in send.sent[-1].message


def test_is_sleeping_handler():
    ctrl, pause, adapter, send, calls = make()
    ctrl.handle_is_sleeping(IsSleepingReqInput())
    assert isinstance(send.sent[-1], IsSleepingReqOutput)
    assert send.sent[-1].is_sleeping is False
