# SPDX-License-Identifier: Apache-2.0
"""Cases, patch sampling and augmentation for the vanilla stack.

A CASE is the unit the pipeline speaks: image (C, K, J, I), label (K, J, I),
an optional per-channel labelled-mask, spacing. Files live as `.npz` — the
simplest honest exchange format; importers for nnU-Net's NIfTI layout belong to
the autonomous entry (T6) and may lazy-import nibabel there, never here.

THE PATCH SAMPLER implements nnU-Net's foreground bias because it works: with
probability `foreground_prob` the patch centre is drawn from the labelled
foreground, otherwise uniformly. Rare structures actually get seen.

AUGMENTATION stays in the family of transforms that cannot invent anatomy:
mirror along any axis and 90-degree in-plane rotation. Resampling-based
augmentation (scale/elastic) belongs to a later step with proper interpolation
rules — shipping it half-defined would be worse than not shipping it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Case:
    """One training case. `mask` is (C, K, J, I) binary — 1 = labelled —
    or None for fully-labelled cases."""

    image: np.ndarray
    label: np.ndarray
    mask: np.ndarray | None
    spacing_mm: tuple[float, float, float]
    case_id: str = ""

    def __post_init__(self) -> None:
        if self.image.ndim != 4:
            raise ValueError(f"image must be (C,K,J,I), got {self.image.shape}")
        if self.label.shape != self.image.shape[1:]:
            raise ValueError(
                f"label {self.label.shape} must match image spatial {self.image.shape[1:]}"
            )
        if self.mask is not None and self.mask.shape != self.image.shape:
            raise ValueError(f"mask {self.mask.shape} must match image {self.image.shape}")


def load_case_npz(
    path: str | Path, spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Case:
    """`image` (float32), `label` (int), optional `mask` (float 0/1)."""
    with np.load(path) as z:
        image = np.asarray(z["image"], dtype=np.float32)
        label = np.asarray(z["label"])
        mask = np.asarray(z["mask"], dtype=np.float32) if "mask" in z.files else None
    return Case(image=image, label=label, mask=mask,
                spacing_mm=spacing_mm, case_id=Path(path).stem)


@dataclass(frozen=True)
class Patch:
    image: np.ndarray
    label: np.ndarray
    mask: np.ndarray | None
    """Centre of the patch in the case's voxel grid — inference and audit
    tooling need to know WHERE a patch came from."""
    centre: tuple[int, int, int]


class PatchSampler:
    """Foreground-biased random patches, with a deterministic seed hook."""

    def __init__(
        self, patch_size: tuple[int, int, int], foreground_prob: float = 1 / 3
    ) -> None:
        if any(p <= 0 for p in patch_size):
            raise ValueError(f"patch sizes are positive, got {patch_size}")
        if not 0.0 <= foreground_prob <= 1.0:
            raise ValueError(f"foreground_prob in [0, 1], got {foreground_prob}")
        self.patch_size = patch_size
        self.foreground_prob = foreground_prob

    def sample(self, case: Case, rng: np.random.Generator) -> Patch:
        shape = case.image.shape[1:]
        centre = self._centre(case, rng)
        sl = self._window(centre, shape)
        return Patch(
            image=case.image[(slice(None),) + sl],
            label=case.label[sl],
            mask=None if case.mask is None else case.mask[(slice(None),) + sl],
            centre=centre,
        )

    def _centre(self, case: Case, rng: np.random.Generator) -> tuple[int, int, int]:
        fg = None
        if rng.random() < self.foreground_prob:
            # Labelled foreground: label > 0 AND, when a mask exists, labelled
            # in it — an annotated case's unlabelled voxels are not foreground
            # for sampling either.
            region = case.label > 0
            if case.mask is not None:
                region = region & (case.mask.max(axis=0) > 0)
            if region.any():
                fg = np.argwhere(region)
        if fg is None or not len(fg):
            return tuple(int(rng.integers(0, n)) for n in case.image.shape[1:])
        return tuple(int(x) for x in fg[rng.integers(0, len(fg))])

    def _window(
        self, centre: tuple[int, int, int], shape: tuple[int, int, int]
    ) -> tuple[slice, ...]:
        slices = []
        for c, n, p in zip(centre, shape, self.patch_size):
            start = min(max(c - p // 2, 0), n - p) if n >= p else 0
            slices.append(slice(start, start + min(p, n)))
        return tuple(slices)


def augment_mirror_rotate(patch: Patch, rng: np.random.Generator) -> Patch:
    """Mirror any subset of axes, then rotate 90 degrees around the I axis.
    Statistics-preserving: no voxel value is ever invented."""
    image, label, mask = patch.image, patch.label, patch.mask
    for axis in (1, 2, 3):
        if rng.random() < 0.5:
            # Image axes are (C,K,J,I); data axis = axis - 1.
            image = np.flip(image, axis=axis - 1)
            label = np.flip(label, axis=axis - 1)
            if mask is not None:
                mask = np.flip(mask, axis=axis - 1)
    k = int(rng.integers(0, 4))
    if k:
        # np.rot90 acts on the last two axes; for the image keep C first.
        image = np.rot90(image, k, axes=(2, 3))
        label = np.rot90(label, k, axes=(1, 2))
        if mask is not None:
            mask = np.rot90(mask, k, axes=(2, 3))
    return Patch(
        image=np.ascontiguousarray(image),
        label=np.ascontiguousarray(label),
        mask=None if mask is None else np.ascontiguousarray(mask),
        centre=patch.centre,
    )


def make_batch(patches: list[Patch]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Stack patches into (B,C,*), (B,*), (B,C,*) — the trainer's batch form."""
    images = np.stack([p.image for p in patches]).astype(np.float32)
    labels = np.stack([p.label for p in patches]).astype(np.int64)
    if any(p.mask is None for p in patches):
        masks = None
    else:
        masks = np.stack([p.mask for p in patches]).astype(np.float32)
    return images, labels, masks
