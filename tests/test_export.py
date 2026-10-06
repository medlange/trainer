# SPDX-License-Identifier: Apache-2.0
"""Exporting a bundle to serving runtimes.

THE CONTRACT UNDER TEST: the exported artefact IS the served net — same
weights, same one-tensor-at-input-resolution semantics — frozen into a form
a serving runtime loads without knowing this tree's vocabulary. Equivalence
is asserted numerically against the PyTorch served net on the same input,
because "the file exists" is the weakest claim that matters.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch
from medos_trainer.vanilla.export import export_bundle
from medos_trainer.vanilla.infer import save_inference_bundle, served_net
from medos_trainer.vanilla.nets import UNetConfig, build_unet

TOY = {"features": (4, 8, 16), "stem_stride": (1, 1, 1)}


def _bundle(tmp_path):
    """A bundle from a random training net — the export path must derive the
    served twin itself, exactly as inference does."""
    torch.manual_seed(0)
    net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                deep_supervision=True, **TOY))
    save_inference_bundle(tmp_path / "bundle", net,
                          {"epoch": 0, "loss": 1.0, "val_masked_dice_loss": 0.0,
                           "lr": 0.01},
                          patch_size=(16, 16, 16))
    return tmp_path / "bundle"


def _served(bundle):
    config = UNetConfig(**json.loads((bundle / "net_config.json").read_text()))
    net = build_unet(config)
    net.load_state_dict(torch.load(bundle / "model.pt", map_location="cpu"))
    return served_net(net).eval()


def _formats() -> tuple[str, ...]:
    """Both formats where the lazy onnx dependency exists, TorchScript alone
    where it does not — the export itself is deliberately usable on a
    torchscript-only install."""
    try:
        import onnx  # noqa: F401
    except ImportError:
        return ("torchscript",)
    return ("torchscript", "onnx")


def test_artifacts_and_manifest_are_written(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    out = tmp_path / "export"
    formats = _formats()
    written = export_bundle(bundle, out, formats=formats)
    for fmt in formats:
        assert written[fmt].is_file()
    manifest = json.loads((out / "export.json").read_text(encoding="utf-8"))
    for fmt in formats:
        entry = manifest[fmt]
        assert entry["format"] == fmt
        assert entry["input_shape"] == [1, 1, 16, 16, 16]
        assert entry["output_shape"] == [1, 2, 16, 16, 16]
    if "onnx" in formats:
        assert manifest["onnx"]["opset"] == 17
    digest = hashlib.sha256((bundle / "model.pt").read_bytes()).hexdigest()
    assert manifest["bundle_digest"] == f"sha256:{digest}"


def test_torchscript_output_matches_the_served_net(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    out = tmp_path / "export"
    export_bundle(bundle, out, formats=("torchscript",))
    # File object, not path: torch.jit.load's C++ path layer fails on
    # parents outside the ANSI codepage (see the note in export.py).
    with open(out / "model.torchscript", "rb") as handle:
        traced = torch.jit.load(handle).eval()
    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        expected = _served(bundle)(x)
        got = traced(x)
    assert isinstance(got, torch.Tensor)
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=2e-4)


def test_onnx_output_matches_the_served_net(tmp_path) -> None:
    ort = pytest.importorskip("onnxruntime")
    bundle = _bundle(tmp_path)
    out = tmp_path / "export"
    export_bundle(bundle, out, formats=("onnx",))
    x = np.random.default_rng(0).normal(size=(1, 1, 16, 16, 16)).astype(np.float32)
    # Bytes, not a path string: the same ANSI-codepage limitation applies to
    # the loader on Windows user profiles outside ASCII.
    session = ort.InferenceSession((out / "model.onnx").read_bytes(),
                                   providers=["CPUExecutionProvider"])
    (got,) = session.run(None, {"input": x})
    with torch.no_grad():
        expected = _served(bundle)(torch.as_tensor(x)).numpy()
    assert got.shape == expected.shape == (1, 2, 16, 16, 16)
    assert np.allclose(got, expected, atol=2e-4)


def test_unknown_format_is_a_lookup_error_with_choices(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    with pytest.raises(ValueError, match="unknown export format"):
        export_bundle(bundle, tmp_path / "export", formats=("tflite",))


def test_export_cli_smoke(tmp_path) -> None:
    from medos_trainer.__main__ import main

    bundle = _bundle(tmp_path)
    out = tmp_path / "export"
    rc = main(["export", "--checkpoint-dir", str(bundle),
               "--out", str(out), "--format", "both"])
    if "onnx" in _formats():
        assert rc == 0
        assert (out / "model.onnx").is_file()
        assert json.loads((out / "export.json").read_text(encoding="utf-8"))
    else:
        # "both" is refused up front, named, and nothing is written; the
        # torchscript-only export still works on the same install.
        assert rc == 3
        assert not (out / "model.torchscript").exists()
        assert not (out / "model.onnx").exists()
        assert main(["export", "--checkpoint-dir", str(bundle),
                     "--out", str(out), "--format", "torchscript"]) == 0
        assert (out / "model.torchscript").is_file()
        assert json.loads((out / "export.json").read_text(encoding="utf-8"))
