# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the cross-build custom-tokenizer loader.

These are pure-Python and need neither a GPU nor tensorrt_llm: the loader imports
tensorrt_llm lazily and the tests mock importlib, so they run in CPU-only CI.
"""

import types
from unittest import mock

import pytest

from dynamo.trtllm.utils.tokenizer_compat import load_custom_tokenizer_compat

pytestmark = [pytest.mark.unit, pytest.mark.trtllm, pytest.mark.pre_merge]

_IMPORT_MODULE = "dynamo.trtllm.utils.tokenizer_compat.importlib.import_module"


def test_prefers_native_loader():
    """When tensorrt_llm.tokenizer.load_custom_tokenizer exists, use it."""
    native = mock.MagicMock(return_value="native-tok")
    native_mod = types.SimpleNamespace(load_custom_tokenizer=native)

    with mock.patch(_IMPORT_MODULE, return_value=native_mod):
        tok = load_custom_tokenizer_compat("glm_moe_dsa", "/models/glm")

    assert tok == "native-tok"
    native.assert_called_once_with("glm_moe_dsa", "/models/glm")


def test_alias_table_dynamic_import_fallback():
    """Without the native helper, the verified alias resolves to its class."""
    tok_cls = mock.MagicMock()
    tok_cls.from_pretrained.return_value = "class-tok"
    native_mod = types.SimpleNamespace()  # no load_custom_tokenizer attribute
    glm_mod = types.SimpleNamespace(GlmMoeDsaTokenizer=tok_cls)

    def fake_import(name):
        return {
            "tensorrt_llm.tokenizer": native_mod,
            "tensorrt_llm.tokenizer.glm_moe_dsa": glm_mod,
        }[name]

    with mock.patch(_IMPORT_MODULE, side_effect=fake_import):
        tok = load_custom_tokenizer_compat("glm_moe_dsa", "/models/glm")

    assert tok == "class-tok"
    tok_cls.from_pretrained.assert_called_once_with("/models/glm")


def test_convention_resolves_bare_alias():
    """A bare snake_case alias (e.g. deepseek_v32) resolves by naming convention."""
    tok_cls = mock.MagicMock()
    tok_cls.from_pretrained.return_value = "ds-tok"
    native_mod = types.SimpleNamespace()
    ds_mod = types.SimpleNamespace(DeepseekV32Tokenizer=tok_cls)

    def fake_import(name):
        return {
            "tensorrt_llm.tokenizer": native_mod,
            "tensorrt_llm.tokenizer.deepseek_v32": ds_mod,
        }[name]

    with mock.patch(_IMPORT_MODULE, side_effect=fake_import):
        tok = load_custom_tokenizer_compat("deepseek_v32", "/models/ds")

    assert tok == "ds-tok"
    tok_cls.from_pretrained.assert_called_once_with("/models/ds")


def test_fully_qualified_path():
    """A dotted name is treated as a module.ClassName import path."""
    tok_cls = mock.MagicMock()
    tok_cls.from_pretrained.return_value = "fq-tok"
    native_mod = types.SimpleNamespace()
    custom_mod = types.SimpleNamespace(CustomTok=tok_cls)

    def fake_import(name):
        return {
            "tensorrt_llm.tokenizer": native_mod,
            "my.custom.module": custom_mod,
        }[name]

    with mock.patch(_IMPORT_MODULE, side_effect=fake_import):
        tok = load_custom_tokenizer_compat("my.custom.module.CustomTok", "/m")

    assert tok == "fq-tok"
    tok_cls.from_pretrained.assert_called_once_with("/m")


def test_unresolvable_alias_raises_runtime_error():
    """A name whose class cannot be imported fails with a precise error."""
    native_mod = types.SimpleNamespace()  # no native loader

    def fake_import(name):
        if name == "tensorrt_llm.tokenizer":
            return native_mod
        raise ImportError("not present in this build")

    with mock.patch(_IMPORT_MODULE, side_effect=fake_import):
        with pytest.raises(RuntimeError, match="could not be imported"):
            load_custom_tokenizer_compat("deepseek_v32", "/m")


def test_forwards_supported_options_to_native_loader():
    """Options accepted by the native loader are forwarded; others are dropped."""
    received = {}

    def fake_native(
        custom_tokenizer, tokenizer_path, trust_remote_code=False, use_fast=True
    ):
        received.update(
            custom_tokenizer=custom_tokenizer,
            tokenizer_path=tokenizer_path,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
        )
        return "tok"

    native_mod = types.SimpleNamespace(load_custom_tokenizer=fake_native)

    with mock.patch(_IMPORT_MODULE, return_value=native_mod):
        tok = load_custom_tokenizer_compat(
            "glm_moe_dsa",
            "/m",
            {"trust_remote_code": True, "tokenizer_mode": "slow", "use_fast": False},
        )

    assert tok == "tok"
    assert received["trust_remote_code"] is True
    assert received["use_fast"] is False
    # tokenizer_mode is not a parameter of fake_native -> dropped, no TypeError.


def test_forwards_options_to_from_pretrained_fallback():
    """tokenizer_mode/trust_remote_code translate to HF from_pretrained kwargs."""
    received = {}

    class FakeTok:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            received.update(path=path, **kwargs)
            return "tok"

    native_mod = types.SimpleNamespace()  # no native loader
    glm_mod = types.SimpleNamespace(GlmMoeDsaTokenizer=FakeTok)

    def fake_import(name):
        return {
            "tensorrt_llm.tokenizer": native_mod,
            "tensorrt_llm.tokenizer.glm_moe_dsa": glm_mod,
        }[name]

    with mock.patch(_IMPORT_MODULE, side_effect=fake_import):
        tok = load_custom_tokenizer_compat(
            "glm_moe_dsa", "/m", {"trust_remote_code": True, "tokenizer_mode": "slow"}
        )

    assert tok == "tok"
    assert received["trust_remote_code"] is True
    assert received["use_fast"] is False  # derived from tokenizer_mode="slow"


def test_derives_use_fast_from_tokenizer_mode_for_native_loader():
    """tokenizer_mode=slow yields use_fast=False even when the native loader takes
    use_fast directly (not tokenizer_mode)."""
    received = {}

    def fake_native(custom_tokenizer, tokenizer_path, use_fast=True):
        received["use_fast"] = use_fast
        return "tok"

    native_mod = types.SimpleNamespace(load_custom_tokenizer=fake_native)

    with mock.patch(_IMPORT_MODULE, return_value=native_mod):
        tok = load_custom_tokenizer_compat(
            "glm_moe_dsa", "/m", {"tokenizer_mode": "slow"}
        )

    assert tok == "tok"
    assert received["use_fast"] is False
