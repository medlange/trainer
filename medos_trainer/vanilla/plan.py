# SPDX-License-Identifier: Apache-2.0
"""Planning: the dataset's fingerprint decides the run's numbers.

nnU-Net's central idea is that nobody should hand-pick a patch size: the
DATA does, through a fingerprint collected once over the training cases.
This module is that idea minus the framework — three functions and two
frozen dataclasses:

    fingerprint = collect_fingerprint(cases)
    plan = plan_from_fingerprint(fingerprint, preset="cpu")
    config, fit_plan = plan.network_config(num_classes=2), plan.fit_plan()

WHAT THE FINGERPRINT HOLDS: median spacing and median shape (the two
numbers every downstream decision reads), the class census over LABELLED
voxels only (an unlabelled voxel is not evidence about rarity), the
labelled fraction (how much of a typical case is actually supervised),
and the foreground intensity statistics (global mean/std over LABELLED
FOREGROUND voxels of channel 0 — the z-score the preprocessor will apply;
single-channel CT is the supported modality).

WHAT THE PLAN DECIDES, and the reasoning the audit can quote:

  * PATCH SIZE from the median spacing. The target is a physical patch of
    about 12 mm through-plane and 20 mm in-plane — the nnU-Net-style
    "context a convolutional net needs" ballpark — converted to voxels per
    axis and rounded to a multiple of 16. Rounding matters: the net pools
    2^(stages-1), and a patch that is not a stride multiple produces
    ragged tails at the bottleneck.
  * STEM STRIDE from anisotropy: when the through-plane spacing is twice
    the in-plane (thick-slice CT), the stem downsamples that axis once —
    the network spends its compute where the resolution is.
  * FEATURES AND BATCH from the VRAM preset. Planning is the only place
    "how much GPU is there" enters; the presets are honest floors, not
    tuned optima, and the plan records which one ran.
  * PATCH CAP: if the spacing-derived patch exceeds the preset's voxel
    budget (isotropic 0.5 mm data makes huge voxel patches), it is
    shrunken axis-wise toward the budget before rounding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

import numpy as np
from medos_trainer.vanilla.data import Case
from medos_trainer.vanilla.nets import UNetConfig
from medos_trainer.vanilla.preprocess import IntensityNorm
from medos_trainer.vanilla.trainer import FitPlan

#: The DEFAULT physical patch target, in millimetres, (through-plane,
#: in-plane, in-plane) — used by presets that do not name their own (today
#: the cpu preset). What counts as "enough anatomy to tell a lesion from a
#: vessel by context" is a function of affordable context, so GPU presets
#: override this with nnU-Net-class targets (see VramPreset). The PulmoAI
#: benchmark found a one-size 12/20/20 mm target handed a 32 GB card an
#: eight-times-smaller patch than nnU-Net plans on the same sub-millimetre
#: CT — that is why this is a preset property, not a global.
TARGET_PATCH_MM: tuple[float, float, float] = (12.0, 20.0, 20.0)

#: Patch axes are rounded to this multiple: five stages pool 16×, and a
#: patch not divisible by 16 either wastes voxels or truncates the deepest
#: feature map, depending on which way the rounding went.
PATCH_ROUNDING = 16


@dataclass(frozen=True)
class Fingerprint:
    """What the training corpus itself says, collected once, frozen after.

    `shapes`/`spacings` keep the full lists so a reviewer can recompute any
    median; the medians are fields (not properties) because the plan is
    frozen against the fingerprint and a property would silently re-derive.
    """

    shapes: tuple[tuple[int, int, int], ...]
    spacings: tuple[tuple[float, float, float], ...]
    median_shape: tuple[int, int, int]
    median_spacing: tuple[float, float, float]
    class_counts: tuple[tuple[int, int], ...]  # (class, labelled voxels)
    labelled_fraction: float
    #: Global mean/std of channel 0 over voxels that are labelled AND
    #: foreground (label > 0, mask-respected like the census) — the
    #: z-score the preprocessor applies. 0.0/0.0 when the corpus carries
    #: no foreground at all; the plan REFUSES that corpus at
    #: `PlannedRun.normalization`, because a corpus without foreground has
    #: no business training.
    foreground_mean: float
    foreground_std: float

    @property
    def num_classes(self) -> int:
        return max(c for c, _ in self.class_counts) + 1

    @property
    def foreground_fraction(self) -> float:
        """Labelled voxels that are not background — the rarity signal."""
        total = sum(n for _, n in self.class_counts)
        if total == 0:
            return 0.0
        background = dict(self.class_counts).get(0, 0)
        return (total - background) / total


@dataclass(frozen=True)
class VramPreset:
    """The one place "which GPU" enters the plan. `voxel_budget` caps the
    patch's voxel count; `features`/`batch_size` are honest CPU/GPU floors,
    `target_patch_mm` is the PHYSICAL context the preset can afford (nnU-Net
    plans ~90–130 mm of context on sub-millimetre CT; a one-size 12/20/20 mm
    target starves a 32 GB card of exactly the context it bought the VRAM
    for — found by the PulmoAI benchmark, see docs/benchmark-pulmo-*.md),
    and the plan records the preset name so a card can say what ran."""

    name: str
    features: tuple[int, ...]
    batch_size: int
    voxel_budget: int
    target_patch_mm: tuple[float, float, float] = TARGET_PATCH_MM


PRESETS: dict[str, VramPreset] = {
    "cpu": VramPreset("cpu", features=(16, 32, 64, 128), batch_size=2,
                      voxel_budget=96 * 96 * 96),
    "small": VramPreset("small", features=(32, 64, 128, 256, 320), batch_size=2,
                        voxel_budget=128 * 128 * 128,
                        target_patch_mm=(40.0, 80.0, 80.0)),
    "large": VramPreset("large", features=(32, 64, 128, 256, 320), batch_size=4,
                        voxel_budget=160 * 160 * 112,
                        target_patch_mm=(80.0, 140.0, 140.0)),
}


@dataclass(frozen=True)
class PlannedRun:
    """The plan: every number the fit needs, and why it is what it is."""

    fingerprint: Fingerprint
    preset: VramPreset
    patch_size: tuple[int, int, int]
    stem_stride: tuple[int, int, int]
    steps_per_epoch: int = 250
    epochs: int = 1000
    learning_rate: float = 0.01
    weight_decay: float = 3e-5
    foreground_prob: float = 1 / 3
    reasons: tuple[str, ...] = field(default=())

    def network_config(self, input_channels: int = 1) -> UNetConfig:
        # RESIDUAL ENCODER BY DEFAULT: the benchmark evidence (see
        # docs/benchmark-pulmo-2026-10-07.md, W16 addendum) is that at real
        # low-contrast CT the plain block optimizes an order of magnitude
        # slower than nnU-Net; the residual encoder is the measured fix
        # (toy learnability 0.23 -> 0.80, test_residual.py). UNetConfig's own
        # default stays False so hand-built nets and old bundles are
        # byte-identical.
        return UNetConfig(
            input_channels=input_channels,
            num_classes=self.fingerprint.num_classes,
            features=self.preset.features,
            stem_stride=self.stem_stride,
            deep_supervision=True,
            residual=True,
        )

    def fit_plan(self) -> FitPlan:
        return FitPlan(
            patch_size=self.patch_size,
            batch_size=self.preset.batch_size,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            foreground_prob=self.foreground_prob,
            # A REAL PLAN gets the full augmentation family: mirror/rotate are
            # free of invention risk, and the scale/elastic pair is the part
            # that teaches scale and deformation invariance. Hand-written
            # FitPlans default it off and keep their byte-exact dynamics.
            augment_resample=True,
            # ... and the INTENSITY tier with them: brightness/contrast/gamma
            # are the nnU-Net-parity transforms the PulmoAI benchmark named as
            # a gap (docs/benchmark-pulmo-2026-10-07.md — nnU-Net augments
            # intensity, we only did geometry). Hand-written FitPlans default
            # it off and keep their byte-exact dynamics.
            augment_intensity=True,
            # and a real plan PREFETCHES: the producer thread overlaps patch
            # sampling/augmentation with the GPU step (see prefetch.py) —
            # the benchmark measured ~0.3-0.5 s/step of CPU work the GPU
            # waited on. Hand-written FitPlans keep 0 (byte-exact, the test
            # suite's reproducible path).
            prefetch_batches=4,
            # POLY, NOT PLATEAU: the budget sweep caught the plateau arm's
            # patience=2 annealing lr to ~0 by epoch ~15 of any long run —
            # 160 epochs scored WORSE than 40 (0.413 vs 0.427) because the
            # run froze. Poly anneals by design to the run's end (nnU-Net's
            # own shape). Hand-written FitPlans keep "plateau" — the defect
            # only bites at real budgets, and the tests pin its arithmetic.
            lr_schedule="poly",
        )

    @property
    def target_spacing(self) -> tuple[float, float, float]:
        """The resampling target: the corpus's own median spacing. The plan
        decides it; `fit_command` resamples training cases forward onto it
        and the bundle's predictor resamples incoming images up to it and
        the probabilities back down, transparently."""
        return self.fingerprint.median_spacing

    @property
    def normalization(self) -> IntensityNorm:
        """The z-score the fingerprint's foreground statistics define.

        REFUSED AT PLAN TIME, NAMED, when the corpus has no foreground
        intensity spread: a corpus without foreground has no business
        training, and a normalization that silently divides by zero would
        produce a worse failure much further from its cause.
        """
        if not self.fingerprint.foreground_std > 1e-8:
            raise ValueError(
                "no intensity normalization: the corpus's foreground "
                f"intensity spread is {self.fingerprint.foreground_std!r} "
                "(foreground_mean="
                f"{self.fingerprint.foreground_mean!r}) — a corpus without "
                "labelled foreground has nothing to segment and no z-score "
                "to train with"
            )
        return IntensityNorm(
            mean=self.fingerprint.foreground_mean,
            std=self.fingerprint.foreground_std,
        )


def collect_fingerprint(cases: list[Case]) -> Fingerprint:
    """The census over the training partition, one pass, no caching subtleties."""
    if not cases:
        raise ValueError("a fingerprint needs at least one case")
    shapes = tuple(tuple(int(v) for v in c.image.shape[1:]) for c in cases)
    spacings = tuple(tuple(float(v) for v in c.spacing_mm) for c in cases)
    counts: dict[int, int] = {}
    labelled = 0
    voxels = 0
    fg_sum = 0.0
    fg_sq_sum = 0.0
    fg_count = 0
    for case in cases:
        label = case.label
        if case.mask is not None:
            # Only voxels the case declares labelled are evidence.
            supervised = case.mask.max(axis=0) > 0
            labelled += int(supervised.sum())
        else:
            supervised = None
            labelled += label.size
        voxels += label.size
        classes = range(int(label.max()) + 1)
        for cls in classes:
            region = label == cls
            if supervised is not None:
                region = region & supervised
            n = int(region.sum())
            if n:
                counts[cls] = counts.get(cls, 0) + n
        # Foreground intensity statistics: channel 0 only (single-channel
        # CT is the supported modality), over voxels that are foreground
        # AND labelled, exactly the census's evidence rule.
        foreground = label > 0
        if supervised is not None:
            foreground = foreground & supervised
        if foreground.any():
            values = case.image[0][foreground]
            fg_sum += float(values.sum(dtype=np.float64))
            fg_sq_sum += float(np.square(values, dtype=np.float64).sum())
            fg_count += int(values.size)
    if not counts:
        raise ValueError("the corpus has no labelled voxels at all")
    if fg_count:
        fg_mean = fg_sum / fg_count
        # E[x^2] - E[x]^2 in float64; clipped at 0 because catastrophic
        # cancellation can leave a breath below zero on a constant field.
        variance = max(fg_sq_sum / fg_count - fg_mean * fg_mean, 0.0)
        fg_std = variance ** 0.5
    else:
        fg_mean, fg_std = 0.0, 0.0
    return Fingerprint(
        shapes=shapes,
        spacings=spacings,
        median_shape=tuple(int(median(axis)) for axis in zip(*shapes)),
        median_spacing=tuple(float(median(axis)) for axis in zip(*spacings)),
        class_counts=tuple(sorted(counts.items())),
        labelled_fraction=labelled / voxels,
        foreground_mean=float(fg_mean),
        foreground_std=float(fg_std),
    )


def _round_to_stride(voxels: float) -> int:
    return max(PATCH_ROUNDING, int(round(voxels / PATCH_ROUNDING)) * PATCH_ROUNDING)


def plan_from_fingerprint(
    fingerprint: Fingerprint, preset: str | VramPreset = "cpu"
) -> PlannedRun:
    """Fingerprint + hardware -> every number the fit needs.

    The reasoning is returned in `reasons` and recorded on the plan, because
    a plan a reviewer cannot interrogate is a number pulled from the air.
    """
    if isinstance(preset, str):
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}: {sorted(PRESETS)}")
        p = PRESETS[preset]
    else:
        p = preset

    spacing = fingerprint.median_spacing
    # The through-plane axis of a medical volume is shapes[0]/spacing[0].
    # target_patch_mm is (through-plane, in-plane, in-plane); medical
    # convention orders spacing the same way, so the pairing is direct.
    # THE TARGET IS A PROPERTY OF THE PRESET, not a global: what counts as
    # "enough context" is affordable context (the benchmark found a global
    # 12/20/20 mm gives a 5090 an eight-times-smaller patch than nnU-Net
    # plans on the same data).
    target = p.target_patch_mm
    voxel_patch = [t / s for t, s in zip(target, spacing)]
    reasons = [
        f"median spacing {spacing} mm -> preset {p.name} target patch "
        f"{target} mm is {tuple(round(v, 1) for v in voxel_patch)} voxels "
        "before rounding",
    ]

    # CAP the voxel budget: high-resolution isotropic data makes enormous
    # voxel patches. Shrink each axis by the same factor until the patch
    # fits, then round — order matters, rounding first can re-overflow.
    voxel_count = voxel_patch[0] * voxel_patch[1] * voxel_patch[2]
    if voxel_count > p.voxel_budget:
        factor = (p.voxel_budget / voxel_count) ** (1.0 / 3.0)
        voxel_patch = [v * factor for v in voxel_patch]
        reasons.append(
            f"voxel count {int(voxel_count)} exceeds preset {p.name}'s budget "
            f"{p.voxel_budget}; shrunk by {factor:.3f}^3",
        )
    patch = tuple(_round_to_stride(v) for v in voxel_patch)

    # STEM STRIDE from anisotropy: through-plane spacing twice the in-plane
    # means the K axis carries half the information per voxel; the stem
    # spends one factor-2 there. In-plane strides stay 1 — organ shape
    # lives in-plane.
    in_plane = (spacing[1] + spacing[2]) / 2.0
    stem = (2, 1, 1) if spacing[0] >= 2.0 * in_plane else (1, 1, 1)
    reasons.append(
        f"through-plane spacing {spacing[0]:.2f} vs in-plane {in_plane:.2f} mm "
        f"-> stem_stride {stem}",
    )

    # PREPROCESSING, the second half of what the plan decides (the
    # benchmark's W16): the median spacing is the resampling target and
    # the foreground intensity census is the z-score. The numbers are
    # recorded here because a plan a reviewer cannot interrogate is a
    # number pulled from the air.
    reasons.append(
        f"foreground intensity over labelled foreground voxels (channel 0): "
        f"mean {fingerprint.foreground_mean:.2f}, std "
        f"{fingerprint.foreground_std:.2f} -> z-score normalization; "
        f"median spacing {spacing} mm is the resampling target (fit "
        "resamples forward, predict resamples back)"
    )

    return PlannedRun(
        fingerprint=fingerprint,
        preset=p,
        patch_size=patch,
        stem_stride=stem,
        reasons=tuple(reasons),
    )
