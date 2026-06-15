# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layer 1 unit tests for the --trtllm-grpc-server frontend flag.

These tests don't bring up a real Dynamo stack; they exercise FrontendConfig
construction and validation only. End-to-end coverage (xp manifest →
SMG-driven Generate stream → token-id round-trip) lands in Layer 3.
"""

import argparse

import pytest

from dynamo.frontend.frontend_args import FrontendConfig


def _base_namespace(**overrides) -> argparse.Namespace:
    """Construct a Namespace with the minimum fields FrontendConfig.validate() reads.

    All other ConfigBase-annotated fields are left to their defaults via
    ConfigBase.from_cli_args's annotation-default fallback.
    """
    defaults = dict(
        tls_cert_path=None,
        tls_key_path=None,
        migration_limit=0,
        kserve_grpc_server=False,
        trtllm_grpc_server=False,
        trtllm_grpc_port=9001,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_default_disables_trtllm_grpc_server():
    cfg = FrontendConfig.from_cli_args(_base_namespace())
    assert cfg.trtllm_grpc_server is False
    assert cfg.kserve_grpc_server is False
    cfg.validate()  # default config validates


def test_trtllm_grpc_server_alone_validates():
    cfg = FrontendConfig.from_cli_args(_base_namespace(trtllm_grpc_server=True))
    assert cfg.trtllm_grpc_server is True
    assert cfg.trtllm_grpc_port == 9001
    cfg.validate()  # OK


def test_kserve_grpc_server_alone_validates():
    cfg = FrontendConfig.from_cli_args(_base_namespace(kserve_grpc_server=True))
    cfg.validate()  # OK


def test_both_grpc_servers_rejected():
    cfg = FrontendConfig.from_cli_args(
        _base_namespace(kserve_grpc_server=True, trtllm_grpc_server=True)
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        cfg.validate()


def test_trtllm_grpc_port_out_of_range_rejected():
    cfg = FrontendConfig.from_cli_args(
        _base_namespace(trtllm_grpc_server=True, trtllm_grpc_port=0)
    )
    with pytest.raises(ValueError, match=r"trtllm-grpc-port"):
        cfg.validate()

    cfg = FrontendConfig.from_cli_args(
        _base_namespace(trtllm_grpc_server=True, trtllm_grpc_port=70000)
    )
    with pytest.raises(ValueError, match=r"trtllm-grpc-port"):
        cfg.validate()


def test_trtllm_grpc_port_only_checked_when_server_enabled():
    """An out-of-range port is fine if the server isn't being started."""
    cfg = FrontendConfig.from_cli_args(
        _base_namespace(trtllm_grpc_server=False, trtllm_grpc_port=0)
    )
    cfg.validate()  # OK — port unused
