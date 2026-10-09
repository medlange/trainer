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
        UNetConfig(input_channels=1, num_classes=2, features=(0, 16))
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
    package's AST and refuse any framework import by name.

    THE GATE READS DEPTH, because the stack's own discipline is depth-based:
    a MODULE-LEVEL import binds every consumer of the package to that
    dependency, so module-level names must be in the plain allowed set; a
    FUNCTION-LEVEL import is the tree's lazy-import pattern (pay nothing for
    what you do not use), and gets its own explicit set. `onnx` is the one
    sanctioned entry there — the ONNX export branch lazy-imports it and raises
    a named RuntimeError without it; an onnx entry at MODULE level, or any
    other lazy name, still fails.
    """
    import ast

    root = Path(__file__).resolve().parents[1] / "medos_trainer" / "vanilla"
    allowed = {"torch", "numpy", "medos_trainer", "dataclasses", "typing",
               "__future__", "math", "enum", "pathlib", "collections",
               "collections.abc", "functools", "itertools", "json", "time",
               "copy", "abc", "io", "statistics",
               # os: vanilla/distributed.py reads WORLD_SIZE/RANK/MASTER_ADDR
               # to decide whether a DDP world exists. Stdlib, same family as
               # pathlib/json/time above — the gate keeps nnU-Net/MONAI out,
               # not the standard library.
               "os",
               # scipy is THIS TREE'S OWN numeric dependency, not a framework:
               # the resampling augmentation (data.py) and the detection/
               # overlap metric modules already use it. The gate's purpose is
               # keeping nnU-Net/MONAI out, not keeping the scientific Python
               # stack out.
               "scipy",
               # hashlib: the export manifest digests model.pt (sha256). Stdlib,
               # deterministic, no framework — same family as json/time above.
               "hashlib",
               # threading + queue: the batch prefetcher's whole mechanism
               # (vanilla/prefetch.py) — ONE daemon producer thread moving
               # sampled/augmented batches through a bounded queue, the
               # stdlib half of torch DataLoader's pin-memory-thread pattern.
               # Stdlib, no framework — the gate keeps nnU-Net/MONAI out.
               "threading", "queue"}
    # THE LAZY SET: function-level imports the optional extras demand. Each
    # entry exists ONLY as `import <name>` inside the one function that needs
    # it, and that function refuses with a named RuntimeError when absent.
    allowed_lazy = {"onnx"}

    class Scanner(ast.NodeVisitor):
        def __init__(self) -> None:
            self.module_level: list[str] = []
            self.function_level: list[str] = []
            self._depth = 0

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._depth += 1
            self.generic_visit(node)
            self._depth -= 1

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._depth += 1
            self.generic_visit(node)
            self._depth -= 1

        def _record(self, names: list[str]) -> None:
            target = self.function_level if self._depth else self.module_level
            target.extend(names)

        def visit_Import(self, node: ast.Import) -> None:
            self._record([a.name.split(".")[0] for a in node.names])

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.module:
                self._record([node.module.split(".")[0]])

    offenders: list[str] = []
    for path in root.rglob("*.py"):
        scanner = Scanner()
        scanner.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        for name in scanner.module_level:
            if name not in allowed:
                offenders.append(f"{path.name} (module): {name}")
        for name in scanner.function_level:
            if name not in allowed | allowed_lazy:
                offenders.append(f"{path.name} (lazy): {name}")
    assert not offenders, f"vanilla stack imports non-vanilla modules: {offenders}"
