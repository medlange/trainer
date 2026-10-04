# SPDX-License-Identifier: Apache-2.0
"""The vanilla networks, held to the same refusal discipline as the rest.

These tests run on CPU with toy shapes; they pin the forward contract that the
data/training/inference slices rely on: deep-supervision tuple ordering, served
single-tensor output, stem-stride behaviour, and the odd-size skip crop.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from medos_trainer.vanilla.nets import UNetConfig, build_unet

TOY = {"features": (8, 16, 32), "stem_stride": (1, 1, 1)}


def test_forward_without_deep_supervision_returns_one_tensor_at_input_resolution() -> None:
    net = build_unet(UNetConfig(input_channels=1, num_classes=3,
                                deep_supervision=False, **TOY))
    out = net(torch.zeros(1, 1, 32, 40, 48))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 3, 32, 40, 48)


def test_deep_supervision_returns_tuple_last_at_input_resolution() -> None:
    net = build_unet(UNetConfig(input_channels=1, num_classes=3,
                                deep_supervision=True, **TOY))
    out = net(torch.zeros(1, 1, 32, 40, 48))
    assert isinstance(out, tuple)
    # One output per decoder stage (pairs = len(features)-1): aux heads on
    # all but the last, the main head on the last.
    assert len(out) == 2
    assert out[-1].shape == (1, 3, 32, 40, 48)
    # Earlier heads sit at downsampled resolutions, same channel count.
    assert out[0].shape[1] == 3
    assert out[0].shape[-3:] == (16, 20, 24)


def test_stem_stride_shrinks_only_where_the_plan_says() -> None:
    net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                deep_supervision=False,
                                features=(8, 16, 32), stem_stride=(2, 2, 1)))
    out = net(torch.zeros(1, 1, 32, 32, 32))
    # stem halves k/j and leaves i alone; the decoder restores each axis it
    # took (two downsamples, two upsamples), so only the stem's axis survive.
    assert out.shape[-3:] == (16, 16, 32)


def test_odd_spatial_size_does_not_crash_the_skip_concat() -> None:
    net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                deep_supervision=False, **TOY))
    out = net(torch.zeros(1, 1, 33, 41, 47))
    assert out.shape == (1, 2, 33, 41, 47)


def test_config_refuses_degenerate_shapes() -> None:
    with pytest.raises(ValueError):
        UNetConfig(input_channels=1, num_classes=2, features=(8,))
    with pytest.raises(ValueError):
        UNetConfig(input_channels=1, num_classes=2, features=(7, 16))
    with pytest.raises(ValueError):
        UNetConfig(input_channels=1, num_classes=2, features=(8, 16), stem_stride=(3, 1, 1))


def test_gradients_flow_to_every_parameter() -> None:
    net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                deep_supervision=True, **TOY))
    out = net(torch.zeros(1, 1, 16, 16, 16))
    loss = sum(o.sum() for o in out)
    loss.backward()
    missing = [name for name, p in net.named_parameters() if p.grad is None]
    assert not missing, f"parameters with no gradient: {missing}"


def test_served_mode_traces_for_export() -> None:
    """The packaging export traces whatever network it is handed; the vanilla
    net must trace and freeze like any other — this is the export contract."""
    net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                deep_supervision=False, **TOY)).eval()
    example = torch.zeros(1, 1, 16, 16, 16)
    with torch.no_grad():
        traced = torch.jit.trace(net, example, strict=False)
        traced = torch.jit.freeze(traced)
        out = traced(example)
    assert out.shape == (1, 2, 16, 16, 16)
    assert isinstance(out, torch.Tensor)


def test_imports_are_vanilla() -> None:
    """THE POINT OF THE PACKAGE: the vanilla stack imports nothing outside the
    stdlib and torch. sys.modules cannot prove this (the suite shares a
    process with the nnU-Net backend tests), so the gate is static: parse the
    package's AST and refuse any framework import by name."""
    import ast

    root = Path(__file__).resolve().parents[1] / "medos_trainer" / "vanilla"
    allowed = {"torch", "dataclasses", "typing", "__future__", "math", "enum",
               "pathlib", "collections", "collections.abc", "functools",
               "itertools", "json", "time", "copy", "abc", "io"}
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name not in allowed:
                    offenders.append(f"{path.name}: {name}")
    assert not offenders, f"vanilla stack imports non-vanilla modules: {offenders}"
