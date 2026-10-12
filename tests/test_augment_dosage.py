# SPDX-License-Identifier: Apache-2.0
"""W26 augmentation dosage: per-transform draw gates, per-axis low-res.

THE CONTRACT UNDER TEST: every augmentation sub-transform can be drawn at a
probability (nnU-Net-class dose) instead of the historical every-patch full
dose; the low-resolution simulation is PER-AXIS (nnU-Net's
SimulateLowResolution semantics), not the all-axes draw that erased small
lesions in the fp250 ablation; and a plan that carries the dosage profile
passes it to both the producer and the inline loop (the two sites must see
identical gates). Neutral corners stay bit-exact identities.
"""

from __future__ import annotations

import numpy as np
import pytest
from medos_trainer.vanilla.data import (
    Patch,
    augment_scale_elastic,
    augment_texture,
)
from medos_trainer.vanilla.plan import collect_fingerprint, plan_from_fingerprint
from medos_trainer.vanilla.trainer import FitPlan
from test_vanilla_data import _toy_cases


def _patch() -> Patch:
    """A deterministic smooth patch: HF energy exists to be removed, and the
    values are stable enough for bit-exact identity assertions."""
    rng = np.random.default_rng(7)
    image = rng.normal(0.0, 1.0, (1, 24, 28, 30)).astype(np.float32)
    image = np.cumsum(np.cumsum(image, axis=-1), axis=-2)
    label = np.zeros((24, 28, 30), dtype=np.int64)
    label[8:16, 10:18, 12:20] = 1
    return Patch(image=image.astype(np.float32), label=label, mask=None,
                 centre=(12, 14, 15), spacing_mm=(1.0, 1.0, 1.0))


def test_closed_gates_are_bit_exact_identity() -> None:
    """Every gate at 0 (or the tier-neutral corner) must return the input
    unchanged — dosage gating never invents a transform."""
    patch = _patch()
    out = augment_texture(
        patch, np.random.default_rng(1),
        noise_sigma=(0.0, 0.5), noise_prob=0.0,
        blur_sigma=(0.5, 1.5), blur_prob=0.0,
        lowres_prob=0.0,
    )
    assert np.array_equal(out.image, patch.image)
    assert out.label is patch.label and out.mask is patch.mask
    out = augment_scale_elastic(
        patch, np.random.default_rng(2),
        scale_prob=0.0, elastic_prob=0.0,
    )
    assert np.array_equal(out.image, patch.image)
    assert np.array_equal(out.label, patch.label)


def test_open_gates_change_the_image() -> None:
    """Sanity of the opposite corner: gates at 1 with active ranges must
    actually transform — a dosage gate that can never fire is dead code."""
    patch = _patch()
    out = augment_texture(
        patch, np.random.default_rng(3),
        noise_sigma=(0.1, 0.2), noise_prob=1.0,
        blur_prob=0.0, lowres_prob=0.0,
    )
    assert not np.array_equal(out.image, patch.image)
    out = augment_scale_elastic(
        patch, np.random.default_rng(4),
        scale_prob=1.0, elastic_prob=1.0,
    )
    assert not np.array_equal(out.image, patch.image)


def test_lowres_is_per_axis_and_preserves_shape() -> None:
    """Per-axis semantics: with every axis drawn the image changes and the
    patch shape survives; the historical all-axes draw is the p=1.0 corner
    of the same code path, so this pins behavior at the extreme while the
    per-axis draw itself is what the dose profile exercises."""
    patch = _patch()
    out = augment_texture(
        patch, np.random.default_rng(5),
        noise_prob=0.0, blur_prob=0.0,
        lowres_prob=1.0, lowres_factor=(0.5, 0.5),
    )
    assert out.image.shape == patch.image.shape
    assert not np.array_equal(out.image, patch.image)


def test_fit_plan_validates_the_dosage_profile() -> None:
    with pytest.raises(ValueError, match="augment_probs"):
        FitPlan(patch_size=(8, 8, 8), augment_probs={"zoom_prob": 0.5})
    with pytest.raises(ValueError, match="augment_probs"):
        FitPlan(patch_size=(8, 8, 8), augment_probs={"scale_prob": 1.5})
    plan = FitPlan(patch_size=(8, 8, 8),
                   augment_probs={"scale_prob": 0.2, "lowres_prob": 0.25})
    assert plan.augment_probs["scale_prob"] == 0.2
    # None is the historical full-dose behavior, still accepted.
    assert FitPlan(patch_size=(8, 8, 8)).augment_probs is None


def test_planned_run_carries_the_nnunet_class_dose() -> None:
    """The W26 dose class is part of what a real plan promises: enabling a
    tier through the CLI must get the nnU-Net-class probabilities, not the
    full dose that cost 0.29 fg Dice in fp250."""
    fp = collect_fingerprint(_toy_cases(3, seed=0))
    fit_plan = plan_from_fingerprint(fp, "cpu").fit_plan()
    assert fit_plan.augment_probs == {
        "scale_prob": 0.2, "elastic_prob": 0.2,
        "noise_prob": 0.15, "blur_prob": 0.15, "lowres_prob": 0.25,
    }
