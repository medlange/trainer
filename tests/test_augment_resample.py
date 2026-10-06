# SPDX-License-Identifier: Apache-2.0
"""The scale/elastic augmentation: shapes, borders, identity, learnability.

THE CONTRACT UNDER TEST mirrors the mirror/rotate tier: patch-shaped output,
no invented anatomy (out-of-range means the patch's own border), discrete
arrays stay discrete — plus the two properties only this tier can claim:
`scale_range=(1.0, 1.0)` with `elastic_alpha_mm=0.0` is EXACTLY the identity
(both branches short-circuit), and turning the augmentation ON must not
break training.
"""

from __future__ import annotations

import numpy as np
import torch
from medos_trainer.vanilla.data import (
    Case,
    PatchSampler,
    augment_scale_elastic,
)
from medos_trainer.vanilla.nets import UNetConfig, build_unet
from medos_trainer.vanilla.trainer import FitPlan, VanillaTrainer


def _case(seed: int = 3) -> Case:
    rng = np.random.default_rng(seed)
    image = rng.normal(-600.0, 50.0, (2, 24, 24, 24)).astype(np.float32)
    label = (rng.random((24, 24, 24)) < 0.08).astype(np.int64)
    mask = np.ones_like(image)
    return Case(image=image, label=label, mask=mask,
                spacing_mm=(2.0, 0.7, 0.7), case_id="aug")


def _patch():
    return PatchSampler((16, 16, 16)).sample(_case(), np.random.default_rng(0))


def test_shapes_preserved_for_image_label_and_mask() -> None:
    patch = _patch()
    aug = augment_scale_elastic(patch, np.random.default_rng(1))
    assert aug.image.shape == patch.image.shape  # (C, 16, 16, 16)
    assert aug.label.shape == patch.label.shape  # (16, 16, 16)
    assert aug.mask.shape == patch.mask.shape
    assert aug.spacing_mm == patch.spacing_mm


def test_scale_at_range_extremes_keeps_shapes_and_discreteness() -> None:
    patch = _patch()
    for factor in (0.85, 1.25):
        aug = augment_scale_elastic(patch, np.random.default_rng(2),
                                    scale_range=(factor, factor),
                                    elastic_alpha_mm=0.0)
        assert aug.image.shape == patch.image.shape, f"zoom {factor}"
        assert set(np.unique(aug.label)) <= {0, 1}
        assert set(np.unique(aug.mask)) <= {0.0, 1.0}


def test_unit_scale_and_zero_elastic_is_the_identity() -> None:
    patch = _patch()
    aug = augment_scale_elastic(patch, np.random.default_rng(3),
                                scale_range=(1.0, 1.0), elastic_alpha_mm=0.0)
    assert np.array_equal(aug.image, patch.image)
    assert np.array_equal(aug.label, patch.label)
    assert np.array_equal(aug.mask, patch.mask)


def test_unit_scale_preserves_the_value_multiset() -> None:
    """Same assertion the mirror/rotate tier carries: statistics-preserving
    transforms do not move a single voxel value."""
    patch = _patch()
    aug = augment_scale_elastic(patch, np.random.default_rng(4),
                                scale_range=(1.0, 1.0), elastic_alpha_mm=0.0)
    assert np.isclose(np.sort(patch.image.ravel()),
                      np.sort(aug.image.ravel())).all()


def test_elastic_alone_preserves_shapes_and_discreteness() -> None:
    patch = _patch()
    aug = augment_scale_elastic(patch, np.random.default_rng(5),
                                scale_range=(1.0, 1.0), elastic_alpha_mm=10.0)
    assert aug.image.shape == patch.image.shape
    assert set(np.unique(aug.label)) <= {0, 1}
    assert set(np.unique(aug.mask)) <= {0.0, 1.0}


def test_training_finds_the_ball_with_resample_augmentation_on_and_off(tmp_path) -> None:
    """The augmentation must TEACH, not just "not crash": both tiers end with
    a net that finds the toy's bright ball on a held-out case. (Val-loss
    improvement alone is too weak a bar — a net that converges to all
    background still improves on background-dominated patches, which is how
    a destructive elastic magnitude slipped past the first version of this
    test.)"""
    from test_vanilla_data import _toy_cases

    train, val = _toy_cases(4, seed=0), _toy_cases(2, seed=100)

    def foreground_dice(case, label) -> float:
        fg = (case.label == 1) & (case.mask[0] > 0)
        denom = float((label == 1).sum() + fg.sum())
        return 2.0 * float((label == 1)[fg].sum()) / denom if denom else 0.0

    def run(augment_resample: bool, out_dir):
        torch.manual_seed(0)
        net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                    features=(4, 8, 16), deep_supervision=True))
        plan = FitPlan(patch_size=(16, 16, 16), batch_size=2, steps_per_epoch=8,
                       epochs=8, foreground_prob=1.0, augment_resample=augment_resample)
        trainer = VanillaTrainer(net, num_classes=2, plan=plan)
        before = trainer.validate(val, np.random.default_rng(9))
        result = trainer.fit(train, val, np.random.default_rng(0), out_dir=out_dir)
        assert result["best_val_masked_dice_loss"] < before, (
            f"augment_resample={augment_resample}: no progress "
            f"({result['best_val_masked_dice_loss']} vs {before})"
        )
        held_out = _toy_cases(1, seed=7)[0]
        label, _ = load_predictor(out_dir).predict(held_out.image)
        return foreground_dice(held_out, label)

    from medos_trainer.vanilla.infer import load_predictor

    off = run(False, tmp_path / "off")
    on = run(True, tmp_path / "on")
    assert off > 0.3, f"unaugmented run did not find the ball: {off}"
    assert on > 0.3, f"resample-augmented run did not find the ball: {on}"
