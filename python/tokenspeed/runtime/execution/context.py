# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool


@dataclass
class ForwardContext:
    """Do not contain Tensor"""

    # --- attention infrastructure ---
    attn_backend: AttentionBackend
    token_to_kv_pool: BaseTokenToKVPool

    # --- meta data ---
    bs: int
    num_extends: int
    input_num_tokens: int
    forward_mode: ForwardMode | None
    req_to_page: torch.Tensor | None = None
    capture_hidden_mode: CaptureHiddenMode | None = CaptureHiddenMode.NULL
    # Spec decode draft head's first step prunes to one live row per request.
    draft_first_step_reduce: bool = False
    # Normalized explicit decode input overrides for this forward, if any.
    decode_input_ids: list[int] | None = None

    # --- dp attention ---
    global_num_tokens: list[int] | None = None
    global_bs: list[int] | None = None
    all_decode_or_idle: bool = False
    # Token range owned by this rank after DP-attention token scatter. These
    # positions are relative to the rank-local DP batch, before TP scattering.
    dp_local_start_pos: int | None = None
    dp_local_num_tokens: int | None = None

    # --- logits processor ---
    gather_ids: torch.Tensor | None = None
    local_gather_ids: torch.Tensor | None = None
    local_gather_positions: torch.Tensor | None = None
    gather_output_size: int | None = None
