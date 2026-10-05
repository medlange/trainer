# SPDX-License-Identifier: Apache-2.0
"""Sliding-window inference for the vanilla stack.

The volume rarely fits the network's patch — a whole CT at 512×512×400 is not
one forward pass. The sliding-window predictor tiles the volume with strided
patches, and where patches overlap it blends with a precomputed Gaussian
weight map: predictions near a patch border (where receptive-field context is
thinnest) contribute less than centre predictions. The weight map is cached
per patch size — recomputing it per window would dominate small volumes.

THE SERVED ARTIFACT IS ONE TENSOR. Training runs with deep supervision (a
tuple of logits); serving runs the same weights with the config flipped to
`deep_supervision=False`, so the served forward returns a single tensor at
input resolution — the contract `packaging.torchscript_bytes` asserts for any
network. `served_net()` builds that twin; `load_predictor()` rebuilds a whole
predictor from a saved bundle.

THIS IS ALSO THE REFERENCE IMPLEMENTATION for the Core-side gap the audit
named: the SDK's kserve_v2 path has no sliding window and no inverse mapping.
When that path is built (C-next), it should read this file, not reinvent it.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from medos_trainer.vanilla.nets import UNetConfig, VanillaUNet


@lru_cache(maxsize=8)
def gaussian_window(
    patch_size: tuple[int, int, int], sigma_scale: float = 0.125
) -> torch.Tensor:
    """The blending weight for one patch, separable 1-D Gaussians.

    `sigma_scale` follows nnU-Net's 1/8 of the patch size: sharp enough that
    windows are mostly independent, wide enough that seams never appear.
    The window is normalised to a max of 1 (a peak, not a pdf): absolute
    calibration across windows is what matters, and every window shares it.
    """
    axes = []
    for p in patch_size:
        sigma = max(p * sigma_scale, 1e-3)
        x = torch.arange(p, dtype=torch.float32) - (p - 1) / 2.0
        g = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
        axes.append(g / g.max())
    return (
        axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    )


def window_starts(length: int, patch: int, overlap: float) -> list[int]:
    """Strided starts with a guaranteed final flush at length - patch.

    Overlap alone leaves a ragged tail uncovered (range stops short); the
    flush window is what makes total coverage exact.
    """
    if length <= patch:
        return [0]
    step = max(int(round(patch * (1.0 - overlap))), 1)
    starts = list(range(0, length - patch + 1, step))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


class SlidingWindowPredictor:
    """Whole-volume inference from a patch network.

    `net` MUST be the served twin (`deep_supervision=False`); a training net
    returns a tuple and the refusal says so, because silently taking the last
    element would bake "which head is the answer" into a place the config
    already answers.
    """

    def __init__(
        self,
        net: VanillaUNet,
        patch_size: tuple[int, int, int],
        overlap: float = 0.5,
        batch_size: int = 2,
        device: str = "cpu",
    ) -> None:
        if net.config.deep_supervision:
            raise ValueError(
                "inference requires the served net (deep_supervision=False): "
                "the artifact predicts one tensor, not a training tuple"
            )
        if not 0.0 <= overlap < 1.0:
            raise ValueError(f"overlap in [0, 1), got {overlap}")
        self.net = net.to(device).eval()
        self.patch_size = patch_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.device = device

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(C,K,J,I) image -> (label (K,J,I) int64, probabilities (C,K,J,I) f32)."""
        x = torch.as_tensor(
            image, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        shape = tuple(x.shape[-3:])
        starts = [
            window_starts(n, p, self.overlap)
            for n, p in zip(shape, self.patch_size)
        ]

        num_classes = self.net.config.num_classes
        probs_sum = torch.zeros(num_classes, *shape, device=self.device)
        weight_sum = torch.zeros(*shape, device=self.device)
        window_weight = gaussian_window(self.patch_size).to(self.device)

        coords = [(k, j, i) for k in starts[0] for j in starts[1] for i in starts[2]]
        for b in range(0, len(coords), self.batch_size):
            batch_coords = coords[b : b + self.batch_size]
            patches = torch.stack(
                [
                    x[
                        0,
                        :,
                        k : k + self.patch_size[0],
                        j : j + self.patch_size[1],
                        i : i + self.patch_size[2],
                    ]
                    for k, j, i in batch_coords
                ]
            )
            out = self.net(patches)
            if not isinstance(out, torch.Tensor):
                raise ValueError(
                    "the served net returned a tuple; build it via served_net() "
                    "so deep_supervision is off"
                )
            probs = F.softmax(out, dim=1)
            for n, (k, j, i) in enumerate(batch_coords):
                sl = (
                    slice(k, k + self.patch_size[0]),
                    slice(j, j + self.patch_size[1]),
                    slice(i, i + self.patch_size[2]),
                )
                probs_sum[(slice(None),) + sl] += probs[n] * window_weight
                weight_sum[sl] += window_weight

        # NO LOWER CLAMP on weight_sum: the flush windows make coverage exact,
        # so every voxel carries a strictly positive weight, and clamping would
        # silently rescale the corners — a 16³ window's corner weight is ~7e-10,
        # below any practical clamp, and the observed bug was corner
        # probabilities off by two orders of magnitude.
        probs = probs_sum / weight_sum.unsqueeze(0)
        return (
            probs.argmax(dim=0).cpu().numpy().astype(np.int64),
            probs.cpu().numpy().astype(np.float32),
        )


def served_net(net: VanillaUNet) -> VanillaUNet:
    """The served twin of a training net: same weights, one output tensor.

    The twin's config flips `deep_supervision` off, so the aux heads do not
    exist in it and `load_state_dict(strict=False)` skips them. The assert
    pins what "skip" may mean: every key the training state has that the twin
    does not is an aux head. A structural change that drops a REAL key is not
    "the aux heads", and must not pass.
    """
    twin = VanillaUNet(
        UNetConfig(
            input_channels=net.config.input_channels,
            num_classes=net.config.num_classes,
            features=net.config.features,
            stem_stride=net.config.stem_stride,
            deep_supervision=False,
        )
    )
    missing, unexpected = twin.load_state_dict(net.state_dict(), strict=False)
    assert not missing, f"served twin is missing weights: {missing}"
    assert all(k.startswith("aux_heads.") for k in unexpected), (
        f"non-aux weights the served twin does not have: {unexpected}"
    )
    return twin


def save_inference_bundle(
    out_dir: str | Path,
    net: VanillaUNet,
    record: dict,
    patch_size: tuple[int, int, int] | None = None,
) -> None:
    """Everything inference needs, beside the checkpoint record itself.

    `model.pt` (training weights, deep supervision included — the served twin
    is derived at load time), `net_config.json` (rebuilds the net),
    `fit_plan.json` with the patch size (not part of `UNetConfig`; without it
    a bundle cannot say what window it was trained on), `checkpoint.json`
    (the caller's record — the trainer writes its best-validation record here).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / "model.pt")
    (out / "net_config.json").write_text(
        json.dumps(asdict(net.config), indent=2), encoding="utf-8"
    )
    if patch_size is not None:
        (out / "fit_plan.json").write_text(
            json.dumps({"patch_size": list(patch_size)}, indent=2),
            encoding="utf-8",
        )
    (out / "checkpoint.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )


def load_predictor(
    checkpoint_dir: str | Path,
    overlap: float = 0.5,
    batch_size: int = 2,
    device: str = "cpu",
) -> SlidingWindowPredictor:
    """Rebuild a predictor from a bundle `save_inference_bundle` wrote.

    The bundle stores the TRAINING net (deep supervision on, aux heads in the
    state dict). Serving goes through `served_net`, so the same weights become
    the one-tensor artifact — one derivation, used by export and by loading.
    """
    out = Path(checkpoint_dir)
    config = UNetConfig(
        **json.loads((out / "net_config.json").read_text(encoding="utf-8"))
    )
    plan = json.loads((out / "fit_plan.json").read_text(encoding="utf-8"))
    net = VanillaUNet(config)
    net.load_state_dict(torch.load(out / "model.pt", map_location=device))
    return SlidingWindowPredictor(
        served_net(net),
        patch_size=tuple(int(v) for v in plan["patch_size"]),
        overlap=overlap,
        batch_size=batch_size,
        device=device,
    )
