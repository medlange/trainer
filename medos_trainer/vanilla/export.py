# SPDX-License-Identifier: Apache-2.0
"""Export a trained bundle to serving runtimes.

A BUNDLE (`model.pt` + `net_config.json` + `fit_plan.json`) is the trainer's
own exchange format; a serving runtime wants a single artefact it can load
without knowing our vocabulary. `export_bundle` rebuilds the SERVED net —
same load path as inference (`infer.served_net` semantics: deep supervision
off, one tensor at input resolution) — and freezes it into:

  * TorchScript, `torch.jit.trace`d on a dummy batch shaped (1, C, *patch)
    read from `fit_plan.json`, the patch size the weights were trained on;
  * ONNX, the classic (dynamo=False) exporter at opset 17 with named
    input/output and a static batch — the serving cost model here is one
    volume at a time, so a dynamic batch dim would be a promise nothing
    in this tree makes use of.

ONNX IS LAZY-IMPORTED inside its branch: torch's exporter needs the `onnx`
package, and a TorchScript-only installation should pay nothing for it —
the RuntimeError names the pip command rather than surfacing a raw
ImportError from three frames deep.

NEXT TO THE ARTIFACTS sits `export.json`: per format the file name, the
input/output shapes and the opset, plus `bundle_digest` — the sha256 of the
bundle's `model.pt`, so a deployment can assert the exported artefact and
the checkpoint it claims to be are one file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from medos_trainer.vanilla.infer import served_net
from medos_trainer.vanilla.nets import UNetConfig, VanillaUNet

#: The ONNX opset the classic exporter targets. 17 is the torch 2.7 default
#: and carries every op this architecture emits (InstanceNorm3d, trilinear
#: Upsample, 3-D conv).
ONNX_OPSET = 17


def _load_served_net(
    checkpoint_dir: str | Path, device: str = "cpu"
) -> tuple[VanillaUNet, tuple[int, int, int]]:
    """THE INFERENCE LOAD PATH, verbatim: training weights in, served twin out.

    The bundle stores the TRAINING net (deep supervision on); serving and
    exporting both derive the one-tensor twin through `served_net`, so the
    exported artefact is bit-identical to what `load_predictor` serves.
    """
    out = Path(checkpoint_dir)
    config = UNetConfig(**json.loads((out / "net_config.json").read_text(encoding="utf-8")))
    plan = json.loads((out / "fit_plan.json").read_text(encoding="utf-8"))
    net = VanillaUNet(config)
    net.load_state_dict(torch.load(out / "model.pt", map_location=device))
    served = served_net(net).to(device).eval()
    patch_size = tuple(int(v) for v in plan["patch_size"])
    return served, patch_size


def export_bundle(
    checkpoint_dir: str | Path,
    out_dir: str | Path,
    formats: tuple[str, ...] | list[str] = ("torchscript", "onnx"),
    device: str = "cpu",
) -> dict[str, Path]:
    """Export `checkpoint_dir`'s served net to each named format, in `out_dir`.

    `formats` entries are "torchscript" and/or "onnx". Returns {format: file}.
    `export.json` next to the artefacts records, per format, file/input shape/
    output shape/opset and the sha256 of the source bundle's `model.pt`.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    served, patch_size = _load_served_net(checkpoint_dir, device=device)
    dummy = torch.zeros(1, served.config.input_channels, *patch_size, device=device)

    manifest: dict[str, dict] = {}
    written: dict[str, Path] = {}
    for fmt in formats:
        if fmt == "torchscript":
            path = out / "model.torchscript"
            with torch.no_grad():
                traced = torch.jit.trace(served, dummy, strict=False)
            # THE FILE OBJECT IS LOAD-BEARING: torch.jit.save's C++ path layer
            # cannot open parents outside the ANSI codepage (a Cyrillic user
            # profile on Windows is enough), while Python's open() handles
            # the same path fine. Same story for the ONNX export below.
            with open(path, "wb") as handle:
                torch.jit.save(traced, handle)
        elif fmt == "onnx":
            path = out / "model.onnx"
            try:
                import onnx  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "exporting to ONNX needs the onnx package: pip install onnx"
                ) from exc
            with open(path, "wb") as handle:
                torch.onnx.export(
                    served,
                    dummy,
                    handle,
                    dynamo=False,
                    opset_version=ONNX_OPSET,
                    input_names=["input"],
                    output_names=["output"],
                )
        else:
            raise ValueError(f"unknown export format {fmt!r}: 'torchscript', 'onnx'")
        with torch.no_grad():
            output_shape = [int(v) for v in served(dummy).shape]
        manifest[fmt] = {
            "format": fmt,
            "file": path.name,
            "input_shape": [int(v) for v in dummy.shape],
            "output_shape": output_shape,
        }
        if fmt == "onnx":
            manifest[fmt]["opset"] = ONNX_OPSET
        written[fmt] = path

    digest = hashlib.sha256(
        (Path(checkpoint_dir) / "model.pt").read_bytes()
    ).hexdigest()
    document = {**manifest, "bundle_digest": f"sha256:{digest}"}
    (out / "export.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    return written


def export_manifest(export_dir: str | Path) -> dict:
    """The export.json `export_bundle` wrote, parsed — the deployment-side
    hook for asserting an artefact's provenance against its bundle."""
    return json.loads(
        (Path(export_dir) / "export.json").read_text(encoding="utf-8")
    )
