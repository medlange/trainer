# SPDX-License-Identifier: Apache-2.0
"""The texture augmentation tier: gaussian noise, gaussian blur, low-res sim.

THE CONTRACT UNDER TEST mirrors the intensity tier's: patch-shaped float
output, label and mask UNTOUCHED (texture is image-only — a single wrong
pixel in the label is a mislabelled voxel), neutral parameters reproduce the
input exactly, and each single-active arm changes a real patch. The payoff
test states the honest contract for this tier: texture ON must still LEARN
on the harder low-contrast toy (noise can hurt a tiny toy, so ON >= OFF is
not asserted — "does not break learning" is the claim, not "always helps").
"""

from __future__ import annotations

import numpy as np
import torch
from medos_trainer.vanilla.data import augment_texture
from medos_trainer.vanilla.infer import load_predictor
from medos_trainer.vanilla.nets import UNetConfig, build_unet
from medos_trainer.vanilla.plan import collect_fingerprint, plan_from_fingerprint
from medos_trainer.vanilla.preprocess import IntensityNorm
from medos_trainer.vanilla.trainer import FitPlan, VanillaTrainer
from test_intensity import _fg_stats, _patch, _toy_cases

# The neutral corner every arm test starts from: nothing active.
_NEUTRAL = {"noise_sigma": (0.0, 0.0), "blur_sigma": (0.0, 0.0), "lowres_prob": 0.0}


def _hf_energy(image: np.ndarray) -> float:
    """Mean |second difference| along every spatial axis — a poor man's
    Laplacian energy. Blur and low-resolution resampling both remove exactly
    the frequency band this measures, so it is the pin for "the image really
    got smoother"."""
    return float(sum(np.abs(np.diff(image, n=2, axis=a)).mean()
                     for a in (-3, -2, -1)))


def test_label_and_mask_pass_through_bit_identical() -> None:
    patch = _patch()
    aug = augment_texture(patch, np.random.default_rng(1))
    # Untouched means untouched: the very same arrays ride along.
    assert aug.label is patch.label
    assert aug.mask is patch.mask
    assert np.array_equal(aug.label, patch.label)
    assert np.array_equal(aug.mask, patch.mask)
    assert aug.image.shape == patch.image.shape
    assert aug.image.dtype == patch.image.dtype
    assert aug.centre == patch.centre
    assert aug.spacing_mm == patch.spacing_mm


def test_neutral_parameters_reproduce_the_input() -> None:
    patch = _patch()
    aug = augment_texture(patch, np.random.default_rng(2), **_NEUTRAL)
    # Every arm is exactly skipped at its neutral corner (sigma 0 adds
    # nothing, blur below 0.3 voxels is skipped, lowres_prob 0 never
    # resamples) — the result is the input, not merely close to it.
    assert np.allclose(aug.image, patch.image, atol=1e-6)


def test_each_single_active_arm_changes_the_image() -> None:
    patch = _patch()
    arms = [
        dict(_NEUTRAL, noise_sigma=(0.05, 0.1)),
        dict(_NEUTRAL, blur_sigma=(0.5, 1.5)),
        dict(_NEUTRAL, lowres_prob=1.0),
    ]
    for arm in arms:
        changed = False
        for seed in (1, 2, 3):  # three chances, against an unlucky draw
            aug = augment_texture(patch, np.random.default_rng(seed), **arm)
            changed |= not np.allclose(aug.image, patch.image)
        assert changed, f"three seeded draws all left the image untouched: {arm}"


def test_blur_reduces_high_frequency_energy() -> None:
    """The blur pin: a LARGE sigma (2.5-3 voxels, kernel ~17 taps) must
    shrink the Laplacian energy below 90% of the original on every seed —
    a "blur" that leaves the second-difference energy unchanged is not a
    blur."""
    patch = _patch()
    before = _hf_energy(patch.image)
    for seed in (1, 2, 3):
        aug = augment_texture(patch, np.random.default_rng(seed),
                              noise_sigma=(0.0, 0.0), blur_sigma=(2.5, 3.0),
                              lowres_prob=0.0)
        after = _hf_energy(aug.image)
        assert after < 0.9 * before, (
            f"blur seed {seed}: HF energy {after:.5f} not below 90% of {before:.5f}"
        )


def test_lowres_prob_zero_is_identity() -> None:
    patch = _patch()
    aug = augment_texture(patch, np.random.default_rng(4),
                          noise_sigma=(0.0, 0.0), blur_sigma=(0.0, 0.0),
                          lowres_prob=0.0, lowres_factor=(0.5, 0.9))
    assert np.allclose(aug.image, patch.image, atol=1e-6)


def test_lowres_reduces_high_frequency_energy() -> None:
    """The low-res pin: at lowres_prob=1.0 the patch must come back changed
    AND smoother — downsampling then trilinearly upsampling back removes
    high-frequency energy on every draw of the 0.5-0.9 factor range."""
    patch = _patch()
    before = _hf_energy(patch.image)
    changed = False
    for seed in (1, 2, 3):
        aug = augment_texture(patch, np.random.default_rng(seed),
                              noise_sigma=(0.0, 0.0), blur_sigma=(0.0, 0.0),
                              lowres_prob=1.0, lowres_factor=(0.5, 0.9))
        changed |= not np.allclose(aug.image, patch.image)
        assert _hf_energy(aug.image) < 0.9 * before, (
            f"lowres seed {seed}: HF energy not below 90% of {before:.5f}"
        )
    assert changed, "three seeded lowres draws all left the image untouched"


def test_training_with_texture_still_learns(tmp_path) -> None:
    """The payoff, at the sibling intensity test's exact configuration
    (features (8,16,32), patch 16^3, fg_prob 1.0, batch 2, 6 epochs x 8
    steps, seed 1) on the harder toy. THE CONTRACT IS "DOES NOT BREAK
    LEARNING", NOT "ALWAYS HELPS": gaussian noise can hurt a tiny toy, so
    both arms — texture ON and OFF — must independently clear a foreground
    Dice floor on the held-out case."""
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
                       epochs=6, foreground_prob=1.0, augment_texture=augment)
        trainer = VanillaTrainer(net, num_classes=2, plan=plan)
        trainer.fit(train, val, np.random.default_rng(1), out_dir=out_dir)
        label, _ = load_predictor(out_dir).predict(held.image)
        return foreground_dice(held, label)

    off = run(False, tmp_path / "off")
    on = run(True, tmp_path / "on")
    print(f"held-out fg dice: texture on={on:.4f} off={off:.4f} "
          f"margin={on - off:+.4f}")
    assert on > 0.4 and off > 0.4, (
        f"texture augmentation broke learning on the low-contrast toy: "
        f"on={on:.4f} off={off:.4f} (floor 0.4)"
    )


def test_planned_run_carries_texture() -> None:
    """A real plan gets the FULL nnU-Net-parity family: geometric
    (mirror/rotate + scale/elastic), intensity, and texture. A hand-written
    FitPlan keeps every default off and stays byte-exact."""
    fp = collect_fingerprint(_toy_cases(3, seed=0))
    fit_plan = plan_from_fingerprint(fp, "cpu").fit_plan()
    assert fit_plan.augment_texture is True
    assert fit_plan.augment_intensity is True
    assert fit_plan.augment_resample is True
    assert FitPlan(patch_size=(16, 16, 16)).augment_texture is False
