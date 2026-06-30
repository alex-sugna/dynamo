# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-build loader for TRT-LLM custom tokenizers.

A checkpoint may declare an SMG-internal ``tokenizer_class`` (e.g. ``glm_moe_dsa``
for GLM-5.2) that stock ``AutoTokenizer`` cannot load. Dynamo must build the
tokenizer the way the TRT-LLM engine does so the worker tokenizer (used for
default sampling params) matches the engine's.

``tensorrt_llm`` is imported lazily inside the functions (never at module load),
so this module — and its tests — import cleanly without a GPU.
"""

import importlib
import inspect

# Verified alias -> (module, class_name) overrides. The native loader resolves
# every alias on builds that ship it; this table only matters on builds that ship
# the tokenizer class but not ``load_custom_tokenizer``. It need only carry
# aliases whose class name does NOT follow the ``<CamelCase>Tokenizer`` convention
# (see ``_resolve_alias``); conventional aliases like ``deepseek_v32`` resolve
# without an entry here.
_CUSTOM_TOKENIZER_ALIASES = {
    "glm_moe_dsa": ("tensorrt_llm.tokenizer.glm_moe_dsa", "GlmMoeDsaTokenizer"),
}


def _resolve_native_custom_tokenizer_loader():
    """Return the native ``load_custom_tokenizer`` helper if present, else None."""
    try:
        module = importlib.import_module("tensorrt_llm.tokenizer")
    except ImportError:
        return None
    return getattr(module, "load_custom_tokenizer", None)


def _resolve_alias(custom_tokenizer):
    """Resolve a ``custom_tokenizer`` name to ``(module_name, class_name)``.

    Order: verified override table; then a fully-qualified ``module.ClassName``
    path (any dotted name); then the TRT-LLM naming convention
    ``tensorrt_llm.tokenizer.<name>.<CamelCase>Tokenizer`` (covers ``glm_moe_dsa``,
    ``deepseek_v32``, and future snake_case aliases).
    """
    if custom_tokenizer in _CUSTOM_TOKENIZER_ALIASES:
        return _CUSTOM_TOKENIZER_ALIASES[custom_tokenizer]
    if "." in custom_tokenizer:
        module_name, class_name = custom_tokenizer.rsplit(".", 1)
        return module_name, class_name
    camel = "".join(part.capitalize() for part in custom_tokenizer.split("_"))
    return f"tensorrt_llm.tokenizer.{custom_tokenizer}", f"{camel}Tokenizer"


def _supported_kwargs(func, options):
    """Subset of ``options`` that ``func`` accepts (all if it takes ``**kwargs``).

    Keeps option forwarding robust across TRT-LLM builds whose loader signatures
    differ, instead of risking a ``TypeError`` on an unexpected keyword.
    """
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(options)
    return {k: v for k, v in options.items() if k in params}


def _hf_from_pretrained_kwargs(options):
    """HF ``from_pretrained`` kwargs: it accepts ``trust_remote_code`` and
    ``use_fast`` but not ``tokenizer_mode`` (``use_fast`` is derived up-front in
    ``load_custom_tokenizer_compat``)."""
    return {
        key: options[key]
        for key in ("trust_remote_code", "use_fast")
        if key in options
    }


def load_custom_tokenizer_compat(
    custom_tokenizer, tokenizer_path, tokenizer_options=None
):
    """Build a custom tokenizer by name, mirroring TRT-LLM's own loading.

      1. Prefer ``tensorrt_llm.tokenizer.load_custom_tokenizer`` when present.
      2. Otherwise import the tokenizer class directly (verified alias, naming
         convention, or fully-qualified ``module.ClassName``).

    ``tokenizer_options`` (e.g. ``trust_remote_code``, ``tokenizer_mode``,
    ``use_fast``) are forwarded so the worker tokenizer matches the engine's.
    Raises a precise error when the implementation is genuinely missing rather
    than silently degrading to ``AutoTokenizer``.
    """
    options = dict(tokenizer_options or {})
    # Derive use_fast from tokenizer_mode up-front so the setting is honored no
    # matter which knob the target accepts — TRT-LLM's native loader (which may
    # take use_fast directly) or HF from_pretrained. tokenizer_mode "slow" maps
    # to use_fast=False (TRT-LLM/vLLM semantics).
    if "use_fast" not in options and "tokenizer_mode" in options:
        options["use_fast"] = options["tokenizer_mode"] != "slow"

    native_loader = _resolve_native_custom_tokenizer_loader()
    if native_loader is not None:
        return native_loader(
            custom_tokenizer,
            tokenizer_path,
            **_supported_kwargs(native_loader, options),
        )

    module_name, class_name = _resolve_alias(custom_tokenizer)
    try:
        module = importlib.import_module(module_name)
        tokenizer_cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            f"custom_tokenizer='{custom_tokenizer}' requires {module_name}."
            f"{class_name}, but it could not be imported ({exc}). Use a "
            "TensorRT-LLM build that includes this tokenizer (or its "
            "load_custom_tokenizer helper), or set custom_tokenizer to a "
            "fully-qualified module.ClassName path."
        ) from exc

    return tokenizer_cls.from_pretrained(
        tokenizer_path, **_hf_from_pretrained_kwargs(options)
    )
