# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MultimodalRequestProcessor.process_openai_request — focused
on the MM-DIAG bisect logging that lets us tell "Rust forwarded the bytes"
apart from "Python worker dropped them".

These tests stub the heavyweight imports (`torch`, `tensorrt_llm`) at
`sys.modules` so the module loads on any host — they do NOT require GPU
or the TRT-LLM runtime.

Run:
    pytest components/src/dynamo/trtllm/tests/test_multimodal_processor_mm_diag.py
"""

from __future__ import annotations

import logging
import sys
import types
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Module-level import shimming. multimodal_processor.py top-level imports
# include `torch` and `tensorrt_llm.llmapi.tokenizer`, which we can't assume
# exist on every test machine. Stub them BEFORE the module is first loaded.
# ---------------------------------------------------------------------------


def _install_stubs() -> None:
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.Tensor = type("Tensor", (), {})  # opaque placeholder
        sys.modules["torch"] = torch_stub

    if "tensorrt_llm" not in sys.modules:
        sys.modules["tensorrt_llm"] = types.ModuleType("tensorrt_llm")
    if "tensorrt_llm.llmapi" not in sys.modules:
        sys.modules["tensorrt_llm.llmapi"] = types.ModuleType("tensorrt_llm.llmapi")
    if "tensorrt_llm.llmapi.tokenizer" not in sys.modules:
        tok_mod = types.ModuleType("tensorrt_llm.llmapi.tokenizer")

        def _tokenizer_factory(*_a, **_kw):
            return None

        tok_mod.tokenizer_factory = _tokenizer_factory
        sys.modules["tensorrt_llm.llmapi.tokenizer"] = tok_mod
    if "tensorrt_llm.inputs" not in sys.modules:
        sys.modules["tensorrt_llm.inputs"] = types.ModuleType("tensorrt_llm.inputs")
    if "tensorrt_llm.inputs.utils" not in sys.modules:
        inputs_utils_mod = types.ModuleType("tensorrt_llm.inputs.utils")

        async def _async_load_image(*_a, **_kw):
            return object()

        inputs_utils_mod.async_load_image = _async_load_image
        sys.modules["tensorrt_llm.inputs.utils"] = inputs_utils_mod

    if "dynamo.common.multimodal.image_loader" not in sys.modules:
        loader_mod = types.ModuleType("dynamo.common.multimodal.image_loader")

        class _ImageLoader:  # opaque placeholder used by type-annotation only
            def __init__(self, *_a, **_kw):
                pass

        loader_mod.ImageLoader = _ImageLoader
        sys.modules["dynamo.common.multimodal.image_loader"] = loader_mod

    if "dynamo.runtime.logging" not in sys.modules:
        log_mod = types.ModuleType("dynamo.runtime.logging")

        def _configure_dynamo_logging():
            pass

        log_mod.configure_dynamo_logging = _configure_dynamo_logging
        sys.modules["dynamo.runtime.logging"] = log_mod


_install_stubs()


# Now safe to import the module under test.
from dynamo.trtllm.multimodal_processor import (  # noqa: E402
    MultimodalRequestProcessor,
)
import dynamo.trtllm.multimodal_processor as mm_processor_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_processor() -> MultimodalRequestProcessor:
    """Construct with a mocked tokenizer + image_loader — neither is touched
    by the early MM-DIAG branch we're testing."""
    proc = MultimodalRequestProcessor.__new__(MultimodalRequestProcessor)
    proc.tokenizer = mock.MagicMock()
    proc.image_loader = mock.MagicMock()
    proc.previous_decoded_text = ""
    proc.mm_handoff_mode = "token_ids"
    return proc


@pytest.fixture
def proc() -> MultimodalRequestProcessor:
    return _make_processor()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_handoff_mode_is_read_once_at_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DYNAMO_TRTLLM_MM_HANDOFF_MODE", "prompt")
    proc = MultimodalRequestProcessor(
        model_type="kimi",
        model_dir="/tmp/model",
        max_file_size_mb=1,
        tokenizer=mock.MagicMock(),
    )
    monkeypatch.setenv("DYNAMO_TRTLLM_MM_HANDOFF_MODE", "token_ids")

    assert proc.mm_handoff_mode == "prompt"


@pytest.mark.asyncio
async def test_mm_diag_logs_when_multi_modal_data_absent(
    proc: MultimodalRequestProcessor, caplog: pytest.LogCaptureFixture
) -> None:
    """Rust handler set multi_modal_data: None → Python side should log
    'no multi_modal_data on request' so we know SMG/Rust didn't forward."""
    request = {"prompt_token_ids": [1, 2, 3]}
    with caplog.at_level(logging.DEBUG):
        await proc.process_openai_request(
            request, embeddings=None, ep_disaggregated_params=None
        )
    msgs = [r.getMessage() for r in caplog.records]
    assert any(
        "MM-DIAG: no multi_modal_data on request" in m for m in msgs
    ), f"missing MM-DIAG absent log; got {msgs}"


@pytest.mark.asyncio
async def test_mm_diag_logs_when_multi_modal_data_present_url_variant(
    proc: MultimodalRequestProcessor, caplog: pytest.LogCaptureFixture
) -> None:
    """Rust handler populated multi_modal_data with the {"Url": "data:..."}
    shape → Python side should log image_url items=1 first_kind=Url and
    the data URL prefix. This is the exact contract the Rust serde shape
    needs to honor for the worker to pick up images."""
    request = {
        "prompt_token_ids": [1, 2, 3],
        "multi_modal_data": {
            "image_url": [
                {"Url": "data:image/jpeg;base64,/9j/AAA"},
            ]
        },
    }
    # The processor will try to load the image via image_loader — short-circuit
    # by raising so the function exits before any heavy decode work. We only
    # care that the MM-DIAG log fired BEFORE that, which it does (it's the
    # first statement in process_openai_request).
    proc.image_loader.load_image_batch = mock.MagicMock(
        side_effect=RuntimeError("short-circuit: diag log already fired")
    )
    with caplog.at_level(logging.DEBUG):
        try:
            await proc.process_openai_request(
                request, embeddings=None, ep_disaggregated_params=None
            )
        except RuntimeError:
            pass  # expected from the mock
    msgs = [r.getMessage() for r in caplog.records]
    diag = next(
        (m for m in msgs if "MM-DIAG: multi_modal_data present" in m), None
    )
    assert diag is not None, f"missing MM-DIAG present log; got {msgs}"
    # Pin the bisect fields so a refactor of the log shape trips the test
    # instead of silently making bisection harder during a real outage.
    assert "image_url items=1" in diag, diag
    assert "first_kind=Url" in diag, diag
    assert "data:image/jpeg;base64,/9j/AAA" in diag, diag


@pytest.mark.asyncio
async def test_mm_diag_logs_when_multi_modal_data_empty_image_url(
    proc: MultimodalRequestProcessor, caplog: pytest.LogCaptureFixture
) -> None:
    """Edge case: dict present but image_url list empty. We still log
    'present' so we can distinguish from 'absent', with items=0."""
    request = {
        "prompt_token_ids": [1, 2, 3],
        "multi_modal_data": {"image_url": []},
    }
    with caplog.at_level(logging.DEBUG):
        await proc.process_openai_request(
            request, embeddings=None, ep_disaggregated_params=None
        )
    msgs = [r.getMessage() for r in caplog.records]
    diag = next(
        (m for m in msgs if "MM-DIAG: multi_modal_data present" in m), None
    )
    assert diag is not None, f"missing MM-DIAG empty log; got {msgs}"
    assert "image_url items=0" in diag, diag


@pytest.mark.asyncio
async def test_image_urls_return_prompt_not_prompt_token_ids(
    proc: MultimodalRequestProcessor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raw image URLs must force TRT-LLM's VLM input processor path.

    If we return prompt_token_ids with multi_modal_data, TRT-LLM keeps the
    already-tokenized tiny prompt and Kimi never expands/runs image tokens.
    """
    image_obj = object()

    async def fake_async_load_image(*_args, **_kwargs):
        return image_obj

    monkeypatch.setattr(mm_processor_mod, "async_load_image", fake_async_load_image)
    proc.mm_handoff_mode = "prompt"
    proc.tokenizer.decode.return_value = (
        "decoded <|media_begin|><|media_pad|><|media_end|> prompt"
    )
    request = {
        "token_ids": [1, 2, 3],
        "multi_modal_data": {
            "image_url": [
                {"Url": "data:image/png;base64,iVBORw0KGgo="},
            ]
        },
    }

    result = await proc.process_openai_request(
        request, embeddings=None, ep_disaggregated_params=None
    )

    assert (
        result["prompt"]
        == "decoded <|media_begin|><|media_pad|><|media_end|> prompt"
    )
    assert result["multi_modal_data"] == {"image": [image_obj]}
    assert "prompt_token_ids" not in result
    proc.tokenizer.decode.assert_called_once_with([1, 2, 3], skip_special_tokens=False)
