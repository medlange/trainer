# SPDX-License-Identifier: Apache-2.0
"""Cases, patch sampling and augmentation for the vanilla stack.

A CASE is the unit the pipeline speaks: image (C, K, J, I), label (K, J, I),
an optional per-channel labelled-mask, spacing. Files live as `.npz` — the
simplest honest exchange format; importers for nnU-Net's NIfTI layout belong to
the autonomous entry (T6) and may lazy-import nibabel there, never here.

THE PATCH SAMPLER implements nnU-Net's foreground bias because it works: with
probability `foreground_prob` the patch centre is drawn from the labelled
foreground, otherwise uniformly. Rare structures actually get seen.

AUGMENTATION has three tiers, and the line between the geometric ones is
"can this invent anatomy": mirror along any axis and 90-degree in-plane
rotation cannot, and apply always; the resampling pair — random zoom and a
smooth elastic warp — can move a border voxel's influence but is built so it
never materialises voxels from nothing (zoom-out pads by REFLECTING the
patch's own border, elastic out-of-range lookups clamp to the nearest real
voxel). Interpolation rules follow the label/mask-are-discrete contract:
image cubic, label and mask nearest. The third tier is INTENSITY —
brightness, contrast, gamma on the image alone — the nnU-Net-parity tier the
PulmoAI benchmark named as a gap (docs/benchmark-pulmo-2026-10-07.md); it
invents nothing and moves nothing, so it runs after the geometric tiers and
leaves label and mask untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.ndimage


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
    """The case's voxel spacing, carried so spacing-aware augmentation (the
    elastic warp's millimetre magnitude) can convert mm to voxels per axis.
    Defaults to isotropic 1 mm — the value `load_case_npz` assumes when the
    `.npz` names no spacing — so a hand-built Patch is never wrong-shaped."""
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)


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
        # FOREGROUND-COORDINATE CACHE: _centre used to run np.argwhere over
        # the whole labelled region on EVERY foreground-biased draw — tens of
        # milliseconds per patch on real CT, paid per training step. The
        # region is a per-case constant, so it is computed once here and
        # keyed by case identity. Keyed by case_id (not id(case)): the fit
        # path resamples cases into NEW objects per run, and id() reuse would
        # serve stale coordinates; case_id is the stable identity within a
        # run. Entries are dropped when a case with the same id re-registers
        # (a re-sampled case recomputes once).
        self._fg_cache: dict[str, np.ndarray | None] = {}

    def sample(self, case: Case, rng: np.random.Generator) -> Patch:
        shape = case.image.shape[1:]
        centre = self._centre(case, rng)
        sl = self._window(centre, shape)
        return Patch(
            image=case.image[(slice(None),) + sl],
            label=case.label[sl],
            mask=None if case.mask is None else case.mask[(slice(None),) + sl],
            centre=centre,
            spacing_mm=case.spacing_mm,
        )

    def _foreground_coords(self, case: Case) -> np.ndarray | None:
        if not case.case_id:
            # Anonymous cases (the default) get no cache entry: the key has
            # no identity, and a collision would serve another case's
            # foreground to this one.
            region = case.label > 0
            if case.mask is not None:
                region = region & (case.mask.max(axis=0) > 0)
            return np.argwhere(region) if region.any() else None
        if case.case_id not in self._fg_cache:
            region = case.label > 0
            if case.mask is not None:
                region = region & (case.mask.max(axis=0) > 0)
            self._fg_cache[case.case_id] = np.argwhere(region) if region.any() else None
        return self._fg_cache[case.case_id]

    def _centre(self, case: Case, rng: np.random.Generator) -> tuple[int, int, int]:
        fg = None
        if rng.random() < self.foreground_prob:
            # Labelled foreground: label > 0 AND, when a mask exists, labelled
            # in it — an annotated case's unlabelled voxels are not foreground
            # for sampling either.
            fg = self._foreground_coords(case)
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

    THE ROTATION IS CONDITIONAL ON J == I. ``np.rot90`` on the (J, I) plane
    swaps those two extents: for a square in-plane patch the shape is
    unchanged, but the plan sizes patches PHYSICALLY (TARGET_PATCH_MM is
    through-plane x in-plane x in-plane), so K routinely differs from J and I
    — an unconditional rotation emitted two different patch shapes into one
    batch and ``make_batch`` refused the stack (found by the PulmoAI
    benchmark; see trainer/docs/benchmark-pulmo-2026-10-06.md). The plan's
    in-plane pair is (20 mm, 20 mm), so every plan-derived patch keeps
    J == I and the rotation still applies exactly where intended; a
    hand-written non-square plan loses the rotation but never the batch.
    Mirrors are per-axis reflections — they never change extents and apply
    unconditionally.

    Statistics-preserving: no voxel value is ever invented.
    """
    image, label, mask = patch.image, patch.label, patch.mask
    for axis in (1, 2, 3):
        if rng.random() < 0.5:
            # Image axes are (C,K,J,I); data axis = axis - 1.
            image = np.flip(image, axis=axis - 1)
            label = np.flip(label, axis=axis - 1)
            if mask is not None:
                mask = np.flip(mask, axis=axis - 1)
    if patch.image.shape[2] == patch.image.shape[3]:
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
        spacing_mm=patch.spacing_mm,
    )


def augment_scale_elastic(
    patch: Patch,
    rng: np.random.Generator,
    scale_range: tuple[float, float] = (0.85, 1.25),
    elastic_alpha_mm: float = 2.0,
    elastic_grid: tuple[int, int, int] = (4, 4, 4),
) -> Patch:
    """Random per-axis zoom, then a smooth elastic warp — patch-shaped output.

    WHY THIS EXISTS: mirroring and rotation teach the net invariance to pose,
    but real imaging varies in SCALE and in soft DEFORMATION. Both are
    resampling operations, and resampling can invent anatomy if it is allowed
    to look outside the patch — so every out-of-range rule here is a border
    rule: zooming out pads by REFLECTING the patch's own border, and the
    elastic warp clamps lookups to the nearest real voxel (`mode="nearest"`).
    Nothing the transform writes came from anywhere but the patch itself.

    THE INTERPOLATION CONTRACT follows the label/mask-are-discrete rule:
    the image warps cubic (order 3), the label and mask nearest (order 0) —
    a fractional class membership or a fractional "labelled" flag would both
    be lies. `elastic_alpha_mm` is a PHYSICAL magnitude, converted to voxels
    through `patch.spacing_mm`, so the same number means the same deformation
    on 0.5 mm CT and 3 mm thick-slice data. THE DEFAULT IS MILD ON PURPOSE:
    the plan's patch is physically ~12-20 mm, and a per-axis displacement std
    beyond a sixth of that scrambles image-label alignment instead of
    bending it — 2 mm deforms, 15 mm destroys. `elastic_grid` is the
    displacement field's coarse lattice (4x4x4 by default); it is upsampled,
    Gaussian-smoothed at sigma = patch/8, then scaled so its per-axis std
    equals the voxel alpha — which also makes `elastic_alpha_mm=0` exactly
    the identity.
    """
    image, label, mask = patch.image, patch.label, patch.mask
    shape = tuple(int(v) for v in image.shape[1:])

    factors = rng.uniform(scale_range[0], scale_range[1], size=3)
    if not np.all(factors == 1.0):
        channel = (1.0,)
        zoom_image = channel + tuple(float(f) for f in factors)
        zoom_label = tuple(float(f) for f in factors)
        # Order 3 for the image, 0 for the discrete arrays — fractional labels
        # would invent classes.
        image = scipy.ndimage.zoom(image, zoom_image, order=3)
        label = scipy.ndimage.zoom(label, zoom_label, order=0)
        if mask is not None:
            mask = scipy.ndimage.zoom(mask, zoom_image, order=0)
        # Recentre on the patch shape: crop the surplus when zoomed in, pad by
        # border reflection when zoomed out — reflection, never constant zero,
        # because a zero pad would teach the net that anatomy fades to nothing
        # at every patch edge.
        image = _crop_or_reflect_pad(image, shape)
        label = _crop_or_reflect_pad(label, shape)
        mask = None if mask is None else _crop_or_reflect_pad(mask, shape)

    if elastic_alpha_mm > 0.0:
        field = _elastic_displacement(shape, patch.spacing_mm, elastic_alpha_mm,
                                      elastic_grid, rng)
        # map_coordinates addresses an (ndim, ...) lattice; the label is 3-D
        # so the field applies directly, the image/mask warp per channel with
        # the same spatial field.
        label = scipy.ndimage.map_coordinates(
            label, field, order=0, mode="nearest", output=label.dtype)
        image = np.stack([
            scipy.ndimage.map_coordinates(
                image[c], field, order=3, mode="nearest").astype(np.float32)
            for c in range(image.shape[0])
        ])
        if mask is not None:
            mask = np.stack([
                scipy.ndimage.map_coordinates(
                    mask[c], field, order=0, mode="nearest", output=np.float32)
                for c in range(mask.shape[0])
            ])

    return Patch(
        image=np.ascontiguousarray(image),
        label=np.ascontiguousarray(label),
        mask=None if mask is None else np.ascontiguousarray(mask),
        centre=patch.centre,
        spacing_mm=patch.spacing_mm,
    )


def augment_intensity(
    patch: Patch,
    rng: np.random.Generator,
    brightness: float = 0.25,
    contrast: tuple[float, float] = (0.65, 1.5),
    gamma: tuple[float, float] = (0.7, 1.5),
) -> Patch:
    """Random brightness shift, contrast scaling and gamma — the image alone.

    WHY THESE THREE: they are nnU-Net's intensity tier (BrightnessTransform /
    ContrastTransform / GammaTransform), the named gap in the PulmoAI
    benchmark (docs/benchmark-pulmo-2026-10-07.md): nnU-Net augments
    intensity, this stack augmented only geometry. Real protocols photograph
    the same pathology at different brightnesses and contrasts; a net that
    only ever sees one intensity regime keys on the regime instead of the
    anatomy — and on a low-contrast corpus that is exactly how foreground
    Dice is lost.

    WHY ON Z-SCORED DATA: this runs AFTER the bundle normalization
    (preprocessing z-scores every case before fit), so `brightness` is a flat
    shift in corpus-standard-deviation units and the contrast/gamma ranges
    bend a histogram the net already sees as O(1) — the same meaning on every
    corpus.

    ORDER MATTERS: the caller runs this AFTER the geometric tiers
    (mirror/rotate, then scale/elastic), never before — geometric
    resampling of a gamma-mapped image would interpolate the mapping, not
    the pixels. Intensity invents nothing and displaces nothing, so unlike
    the geometric tiers it cannot break image-label alignment — which is why
    label and mask pass through UNTOUCHED: a single wrong pixel in the label
    is a mislabelled voxel. Brightness is one flat draw per patch; contrast
    stretches each channel around its own patch mean with a per-channel
    factor; gamma per channel maps a (p1, p99) percentile window to [0, 1],
    bends it with the power law, and maps back — a degenerate window
    (p99 - p1 < 1e-6, a constant channel) skips gamma for that channel
    rather than dividing by zero. The value math runs in float64 and casts
    back, so neutral parameters reproduce the input to within a float32
    round-off — far inside any training tolerance.
    """
    image = patch.image.copy()
    # BRIGHTNESS: one flat draw, added to every voxel of every channel.
    image += np.float32(rng.uniform(-brightness, brightness))
    # CONTRAST: stretch each channel around its own patch mean. The mean is
    # the level the net has learned to expect, so the channel pivots there.
    for c in range(image.shape[0]):
        channel = image[c].astype(np.float64)
        mean = float(channel.mean())
        channel = (channel - mean) * float(rng.uniform(*contrast)) + mean
        # GAMMA: the percentile window is where the channel's information
        # actually lives; the power law bends INSIDE it and the linear map
        # restores the scale. Voxels OUTSIDE the window (the ~2% tails —
        # a bright vessel, metal) pass through untouched: gamma must not
        # crush them, and leaving them out keeps neutral parameters at a
        # float32-round-trip from the identity. `inside` also keeps a
        # fractional power away from negative bases.
        p1, p99 = np.percentile(channel, (1.0, 99.0))
        window = float(p99 - p1)
        if window >= 1e-6:
            g = float(rng.uniform(*gamma))
            inside = (channel >= p1) & (channel <= p99)
            norm = (channel[inside] - p1) / window
            channel[inside] = norm ** g * window + p1
        image[c] = channel.astype(image.dtype)
    return Patch(
        image=np.ascontiguousarray(image, dtype=patch.image.dtype),
        label=patch.label,
        mask=patch.mask,
        centre=patch.centre,
        spacing_mm=patch.spacing_mm,
    )


def _crop_or_reflect_pad(arr: np.ndarray, target: tuple[int, int, int]) -> np.ndarray:
    """Bring the spatial axes of `arr` to `target`: central crop when larger,
    border-reflection pad when smaller. Channel axes ride along untouched."""
    out = arr
    spatial = range(out.ndim - 3, out.ndim)
    for axis, t in zip(spatial, target):
        n = out.shape[axis]
        if n > t:
            start = (n - t) // 2
            out = np.take(out, indices=range(start, start + t), axis=axis)
        elif n < t:
            left = (t - n) // 2
            right = t - n - left
            pads = [(0, 0)] * out.ndim
            pads[axis] = (left, right)
            out = np.pad(out, pads, mode="reflect")
    return out


def _elastic_displacement(
    shape: tuple[int, int, int],
    spacing_mm: tuple[float, float, float],
    alpha_mm: float,
    grid: tuple[int, int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """(3, K, J, I) lookup coordinates: integer lattice plus a smoothed,
    magnitude-calibrated displacement field."""
    coarse = rng.normal(0.0, 1.0, (3, *grid)).astype(np.float32)
    zooms = (1.0,) + tuple(n / g for n, g in zip(shape, grid))
    field = scipy.ndimage.zoom(coarse, zooms, order=3)
    coords = np.empty((3, *shape), dtype=np.float32)
    for axis, n in enumerate(shape):
        smoothed = scipy.ndimage.gaussian_filter(field[axis], sigma=max(n / 8.0, 1e-3))
        std = float(smoothed.std())
        if std > 0.0:
            # Calibrate the magnitude: per-axis std == alpha in VOXELS, with mm
            # converted through the spacing. Scaling (not clipping) keeps the
            # field smooth and zero-mean.
            smoothed = smoothed * (alpha_mm / spacing_mm[axis] / std)
        lattice = np.arange(n, dtype=np.float32)
        coords[axis] = lattice.reshape(
            (-1,) + (1,) * (len(shape) - 1 - axis)) + smoothed
    return coords


def make_batch(patches: list[Patch]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Stack patches into (B,C,*), (B,*), (B,C,*) — the trainer's batch form."""
    images = np.stack([p.image for p in patches]).astype(np.float32)
    labels = np.stack([p.label for p in patches]).astype(np.int64)
    if any(p.mask is None for p in patches):
        masks = None
    else:
        masks = np.stack([p.mask for p in patches]).astype(np.float32)
    return images, labels, masks
