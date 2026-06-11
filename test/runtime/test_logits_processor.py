"""Regression tests for logits processing helpers."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, suite="runtime-1gpu")

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import CaptureHiddenMode, ForwardMode
from tokenspeed.runtime.layers import logits_processor as logits_processor_module
from tokenspeed.runtime.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    fused_softcap,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_softcap_handles_large_logits_without_nan():
    cap = 30.0
    logits = torch.tensor(
        [[5000.0, 2000.0, 1500.0, 100.0, 0.0, -100.0, -1500.0, -5000.0]],
        device="cuda",
        dtype=torch.float32,
    )
    expected = cap * torch.tanh(logits / cap)

    out = fused_softcap(logits.clone(), cap)
    torch.cuda.synchronize()

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=2e-5)


def test_dp_attention_empty_local_hidden_returns_empty_logits_without_lm_head_launch(
    monkeypatch,
):
    processor = LogitsProcessor(
        SimpleNamespace(vocab_size=7, model_type="kimi_k2"),
        skip_all_gather=True,
        tp_rank=0,
        tp_size=4,
        tp_group=(0, 1, 2, 3),
    )
    hidden_states = torch.empty((0, 3), dtype=torch.float32)
    lm_head = SimpleNamespace(weight=torch.empty((11, 3), dtype=torch.bfloat16))

    def fail_on_fused_lookup():
        raise AssertionError("empty DP-attention hidden must not launch fused lm_head")

    monkeypatch.setattr(
        logits_processor_module,
        "_get_fused_lm_head_gemm",
        fail_on_fused_lookup,
    )

    logits = processor._get_logits(
        hidden_states,
        lm_head,
        LogitsMetadata(forward_mode=ForwardMode.DECODE),
    )

    assert logits.shape == (0, 7)
    assert logits.dtype == torch.bfloat16


def test_empty_hidden_single_tp_returns_empty_logits():
    processor = LogitsProcessor(
        SimpleNamespace(vocab_size=7, model_type="unit_test"),
        tp_rank=0,
        tp_size=1,
    )
    hidden_states = torch.empty((0, 3), dtype=torch.float32)
    lm_head = SimpleNamespace(weight=torch.empty((11, 3), dtype=torch.float16))

    logits = processor._get_logits(
        hidden_states,
        lm_head,
        LogitsMetadata(forward_mode=ForwardMode.DECODE),
    )

    assert logits.shape == (0, 7)
    assert logits.dtype == torch.float16


def test_empty_local_hidden_skips_global_gather_ids_before_lm_head(monkeypatch):
    processor = LogitsProcessor(
        SimpleNamespace(vocab_size=7, model_type="kimi_k2"),
        skip_all_gather=True,
        tp_rank=0,
        tp_size=4,
        tp_group=(0, 1, 2, 3),
    )
    hidden_states = torch.empty((0, 3), dtype=torch.float32)
    metadata = LogitsMetadata(
        forward_mode=ForwardMode.EXTEND,
        gather_ids=torch.tensor([6], dtype=torch.int64),
    )
    lm_head = SimpleNamespace(weight=torch.empty((11, 3), dtype=torch.bfloat16))
    seen = {}

    def fake_get_logits(pruned_states, *_args, **_kwargs):
        seen["shape"] = pruned_states.shape
        return torch.empty((0, 7), dtype=torch.bfloat16)

    monkeypatch.setattr(processor, "_get_logits", fake_get_logits)

    output = processor.forward(
        input_ids=None,
        hidden_states=hidden_states,
        lm_head=lm_head,
        logits_metadata=metadata,
    )

    assert seen["shape"] == (0, 3)
    assert output.next_token_logits.shape == (0, 7)


def test_dp_local_hidden_drops_global_gather_ids_owned_by_other_rank(monkeypatch):
    processor = LogitsProcessor(
        SimpleNamespace(vocab_size=7, model_type="kimi_k2"),
        skip_all_gather=True,
        tp_rank=0,
        tp_size=4,
        tp_group=(0, 1, 2, 3),
    )
    hidden_states = torch.arange(2 * 3, dtype=torch.float32).view(2, 3)
    metadata = LogitsMetadata(
        forward_mode=ForwardMode.EXTEND,
        gather_ids=torch.tensor([6], dtype=torch.int64),
        dp_local_start_pos=0,
        dp_local_num_tokens=2,
    )
    lm_head = SimpleNamespace(weight=torch.empty((11, 3), dtype=torch.bfloat16))
    seen = {}

    def fake_get_logits(pruned_states, *_args, **_kwargs):
        seen["shape"] = pruned_states.shape
        return torch.empty((0, 7), dtype=torch.bfloat16)

    monkeypatch.setattr(processor, "_get_logits", fake_get_logits)

    output = processor.forward(
        input_ids=None,
        hidden_states=hidden_states,
        lm_head=lm_head,
        logits_metadata=metadata,
    )

    assert seen["shape"] == (0, 3)
    assert output.next_token_logits.shape == (0, 7)


def test_dp_local_hidden_maps_global_gather_ids_to_owner_rank(monkeypatch):
    processor = LogitsProcessor(
        SimpleNamespace(vocab_size=7, model_type="kimi_k2"),
        skip_all_gather=True,
        tp_rank=3,
        tp_size=4,
        tp_group=(0, 1, 2, 3),
    )
    hidden_states = torch.arange(3, dtype=torch.float32).view(1, 3)
    metadata = LogitsMetadata(
        forward_mode=ForwardMode.EXTEND,
        gather_ids=torch.tensor([6], dtype=torch.int64),
        dp_local_start_pos=6,
        dp_local_num_tokens=1,
    )
    lm_head = SimpleNamespace(weight=torch.empty((11, 3), dtype=torch.bfloat16))
    seen = {}

    def fake_get_logits(pruned_states, *_args, **_kwargs):
        seen["states"] = pruned_states
        return torch.empty((1, 7), dtype=torch.bfloat16)

    monkeypatch.setattr(processor, "_get_logits", fake_get_logits)

    output = processor.forward(
        input_ids=None,
        hidden_states=hidden_states,
        lm_head=lm_head,
        logits_metadata=metadata,
    )

    assert torch.equal(seen["states"], hidden_states)
    assert output.next_token_logits.shape == (1, 7)


def test_logits_metadata_prefers_local_gather_ids_from_forward_context():
    global_gather_ids = torch.tensor([6], dtype=torch.int64)
    local_gather_ids = torch.empty((0,), dtype=torch.int64)
    ctx = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        gather_ids=global_gather_ids,
        local_gather_ids=local_gather_ids,
        dp_local_start_pos=0,
        dp_local_num_tokens=2,
    )

    metadata = LogitsMetadata.from_forward_context(ctx)

    assert metadata.gather_ids is local_gather_ids
    assert metadata.dp_local_start_pos is None
    assert metadata.dp_local_num_tokens is None
