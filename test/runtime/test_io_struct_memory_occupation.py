# SPDX-License-Identifier: Apache-2.0
from tokenspeed.runtime.engine.io_struct import (
    IsSleepingReqInput,
    IsSleepingReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
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
