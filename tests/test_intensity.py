# SPDX-License-Identifier: Apache-2.0
"""The intensity augmentation: image-only contract, identity pins, learnability.

THE CONTRACT UNDER TEST mirrors the geometric tiers where it can: patch-shaped
float output, label and mask UNTOUCHED (intensity is an image-only transform —
a single wrong pixel in the label is a mislabelled voxel), neutral parameters
reproduce the input exactly, and each single-active arm changes a real patch.
The payoff test is the one only this tier can claim: on a toy whose classes
overlap the way low-contrast real CT does, turning intensity augmentation ON
must beat OFF on held-out foreground Dice — a transform that only "does not
crash" is decoration.
"""

from __future__ import annotations

import numpy as np
import torch
from medos_trainer.vanilla.data import Case, PatchSampler, augment_intensity
from medos_trainer.vanilla.infer import load_predictor
from medos_trainer.vanilla.nets import UNetConfig, build_unet
from medos_trainer.vanilla.plan import collect_fingerprint, plan_from_fingerprint
from medos_trainer.vanilla.preprocess import IntensityNorm
from medos_trainer.vanilla.trainer import FitPlan, VanillaTrainer


def _case(seed: int = 3) -> Case:
    """A raw CT-scale case: background noise plus structure to segment."""
    rng = np.random.default_rng(seed)
    image = rng.normal(-600.0, 50.0, (2, 24, 24, 24)).astype(np.float32)
    label = (rng.random((24, 24, 24)) < 0.08).astype(np.int64)
    mask = np.ones_like(image)
    return Case(image=image, label=label, mask=mask,
                spacing_mm=(2.0, 0.7, 0.7), case_id="aug")


def _patch():
    """A sampled patch at the transform's operating point: the case z-scored,
    as preprocessing leaves every case before fit."""
    case = _case()
    z = (case.image - float(case.image.mean())) / float(case.image.std())
    case = Case(image=z.astype(np.float32), label=case.label, mask=case.mask,
                spacing_mm=case.spacing_mm, case_id=case.case_id)
    return PatchSampler((16, 16, 16)).sample(case, np.random.default_rng(0))


def test_label_and_mask_pass_through_bit_identical() -> None:
    patch = _patch()
    aug = augment_intensity(patch, np.random.default_rng(1))
    # Untouched means untouched: the very same arrays ride along.
    assert aug.label is patch.label
    assert aug.mask is patch.mask
    assert np.array_equal(aug.label, patch.label)
    assert np.array_equal(aug.mask, patch.mask)
    assert aug.image.shape == patch.image.shape
    assert aug.image.dtype == patch.image.dtype
    assert aug.centre == patch.centre
    assert aug.spacing_mm == patch.spacing_mm


def test_image_changes_with_probability_one() -> None:
    patch = _patch()
    changed = False
    for seed in (1, 2, 3):  # three chances, against an astronomically unlucky draw
        aug = augment_intensity(patch, np.random.default_rng(seed))
        changed |= not np.allclose(aug.image, patch.image)
    assert changed, "three seeded draws all left the image untouched"


def test_neutral_parameters_reproduce_the_input() -> None:
    patch = _patch()
    aug = augment_intensity(patch, np.random.default_rng(4),
                            brightness=0.0, contrast=(1.0, 1.0), gamma=(1.0, 1.0))
    # Neutral parameters are the identity up to a float32 round-off: with the
    # float64 internals the residual is at most a 1-ULP flip on boundary
    # voxels, orders of magnitude inside this tolerance.
    assert np.allclose(aug.image, patch.image, atol=1e-6)


def test_each_single_active_arm_changes_the_image() -> None:
    patch = _patch()
    arms = [
        {"brightness": 0.25, "contrast": (1.0, 1.0), "gamma": (1.0, 1.0)},
        {"brightness": 0.0, "contrast": (0.65, 1.5), "gamma": (1.0, 1.0)},
        {"brightness": 0.0, "contrast": (1.0, 1.0), "gamma": (0.7, 1.5)},
    ]
    for arm in arms:
        aug = augment_intensity(patch, np.random.default_rng(5), **arm)
        assert not np.allclose(aug.image, patch.image), (
            f"single active arm left the image untouched: {arm}"
        )


def test_gamma_degenerate_window_guard() -> None:
    """A constant channel has no window to bend: gamma must SKIP it, not
    divide by zero — with gamma the only active arm the image comes back
    exactly as it went in."""
    case = _case(seed=8)
    flat = np.full_like(case.image, -1.75)
    patch = PatchSampler((16, 16, 16)).sample(
        Case(image=flat, label=case.label, mask=case.mask,
             spacing_mm=case.spacing_mm, case_id="flat"),
        np.random.default_rng(0))
    aug = augment_intensity(patch, np.random.default_rng(6),
                            brightness=0.0, contrast=(1.0, 1.0), gamma=(0.7, 1.5))
    assert np.array_equal(aug.image, patch.image)


def _toy_cases(n: int = 4, seed: int = 0) -> list[Case]:
    """The _toy_cases pattern made harder: every case photographs the sphere
    differently — its intensity scaled by U(0.5, 1.5), the whole image biased
    by U(-1, 1) — BEFORE the corpus z-score, so after normalization the
    foreground no longer sits at one canonical intensity. That is the
    low-contrast regime the PulmoAI benchmark measured (fg Dice lost to
    intensity memorization), in toy form."""
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n):
        image = rng.normal(-600.0, 50.0, (1, 24, 24, 24)).astype(np.float32)
        label = np.zeros((24, 24, 24), dtype=np.int64)
        c = rng.integers(8, 16, size=3)
        kk, jj, ii = np.ogrid[:24, :24, :24]
        ball = (kk - c[0]) ** 2 + (jj - c[1]) ** 2 + (ii - c[2]) ** 2 <= 5 ** 2
        image[0][ball] = 200.0 * float(rng.uniform(0.5, 1.5))
        image += float(rng.uniform(-1.0, 1.0))
        label[ball] = 1
        mask = np.ones_like(image)
        mask[:, :12, :12, :12] = 0.0
        cases.append(Case(image=image, label=label, mask=mask,
                          spacing_mm=(1.0, 1.0, 1.0), case_id=f"toy-{i}"))
    return cases


def test_training_with_intensity_beats_without(tmp_path) -> None:
    """The payoff, at the sibling test's exact configuration (features
    (8,16,32), patch 16^3, fg_prob 1.0, batch 2, 6 epochs x 8 steps, seed 1)
    on the harder toy: intensity ON must end STRICTLY ahead of OFF on
    held-out foreground Dice. Identical seeds and nets — the augmentation
    flag is the only difference, so the comparison is the tier itself."""
    train_raw = _toy_cases(4, seed=0)
    # The corpus z-score the fingerprint would freeze, over the training
    # partition only: the held-out case must be normalized with statistics
    # it could not have contributed to.
    norm = IntensityNorm(*_fg_stats(train_raw))
    train = [norm.apply(c) for c in train_raw]
    val = [norm.apply(c) for c in _toy_cases(2, seed=100)]
    held = norm.apply(_toy_cases(1, seed=7)[0])

    def foreground_dice(case, label) -> float:
        fg = (case.label == 1) & (case.mask[0] > 0)
        denom = float((label == 1).sum() + fg.sum())
        return 2.0 * float((label == 1)[fg].sum()) / denom if denom else 0.0

    def run(augment: bool, out_dir):
        torch.manual_seed(1)
        net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                    features=(8, 16, 32), deep_supervision=True))
        plan = FitPlan(patch_size=(16, 16, 16), batch_size=2, steps_per_epoch=8,
                       epochs=6, foreground_prob=1.0, augment_intensity=augment)
        trainer = VanillaTrainer(net, num_classes=2, plan=plan)
        trainer.fit(train, val, np.random.default_rng(1), out_dir=out_dir)
        label, _ = load_predictor(out_dir).predict(held.image)
        return foreground_dice(held, label)

    off = run(False, tmp_path / "off")
    on = run(True, tmp_path / "on")
    print(f"held-out fg dice: intensity on={on:.4f} off={off:.4f} "
          f"margin={on - off:+.4f}")
    assert on > off, (
        f"intensity augmentation did not pay on the low-contrast toy: "
        f"on={on:.4f} off={off:.4f}"
    )


def _fg_stats(cases: list[Case]) -> tuple[float, float]:
    """The corpus-level foreground z-score the fingerprint would freeze."""
    fp = collect_fingerprint(cases)
    return fp.foreground_mean, fp.foreground_std


def test_planned_run_carries_intensity() -> None:
    """A real plan gets the full augmentation family: the INTENSITY tier
    (pure numpy) and the RESAMPLING pair (torch-native since the W18 fix).
    A hand-written FitPlan keeps every default off and stays byte-exact."""
    fp = collect_fingerprint(_toy_cases(3, seed=0))
    fit_plan = plan_from_fingerprint(fp, "cpu").fit_plan()
    assert fit_plan.augment_intensity is True
    assert fit_plan.augment_resample is True
    assert FitPlan(patch_size=(16, 16, 16)).augment_intensity is False
