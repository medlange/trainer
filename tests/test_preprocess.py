# SPDX-License-Identifier: Apache-2.0
"""Preprocessing parity with nnU-Net: resampling + foreground z-score.

The PulmoAI benchmark (docs/benchmark-pulmo-2026-10-07.md) measured the
cost of their absence — nnU-Net 0.764 vs 0.000 foreground Dice at five
epochs — and this module is the closure. What is under test here is the
CONTRACT, not the numerics of cubic interpolation:

  * the fingerprint's foreground statistics are the global mean/std over
    LABELLED FOREGROUND voxels of channel 0, mask-respected like the census;
  * z-scoring with them makes the foreground standard normal, and a
    degenerate std is REFUSED, NAMED, never silently divided by;
  * resampling carries spacing onto the case, keeps the label set closed
    under nearest-neighbour, and round-trips a smooth image;
  * the plan records the decision (reason line) and refuses a corpus with
    no foreground intensity spread;
  * the fit path writes `preprocess.json` into the bundle, the predict
    path replays it transparently and returns the label on the INPUT
    grid — including for a case whose spacing differs from the training
    corpus's median.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from medos_trainer.standalone import fit_command
from medos_trainer.vanilla.data import Case
from medos_trainer.vanilla.infer import load_predictor, save_inference_bundle
from medos_trainer.vanilla.nets import UNetConfig, build_unet
from medos_trainer.vanilla.plan import collect_fingerprint, plan_from_fingerprint
from medos_trainer.vanilla.preprocess import (
    IntensityNorm,
    Preprocessing,
    preprocess_cases,
    resample,
    resample_case,
)


def _sphere_case(
    shape: tuple[int, int, int] = (24, 24, 24),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    foreground_value: float = 200.0,
    seed: int = 0,
    unlabel_a_corner: bool = True,
) -> Case:
    """The toy the vanilla suite shares: a bright sphere is class 1, the
    background is -600-ish, a corner quadrant may be unlabelled."""
    rng = np.random.default_rng(seed)
    image = rng.normal(-600.0, 50.0, (1, *shape)).astype(np.float32)
    label = np.zeros(shape, dtype=np.int64)
    c = rng.integers(8, 16, size=3)
    kk, jj, ii = np.ogrid[: shape[0], : shape[1], : shape[2]]
    ball = (kk - c[0]) ** 2 + (jj - c[1]) ** 2 + (ii - c[2]) ** 2 <= 5 ** 2
    # Real foregrounds have texture: a perfectly constant sphere would have
    # zero intensity spread and the plan would refuse to normalize it.
    image[0][ball] = foreground_value + rng.normal(0.0, 25.0, int(ball.sum()))
    label[ball] = 1
    mask = np.ones_like(image)
    if unlabel_a_corner:
        mask[:, : shape[0] // 2, : shape[1] // 2, : shape[2] // 2] = 0.0
    return Case(image=image, label=label, mask=mask,
                spacing_mm=spacing, case_id=f"sphere-{seed}")


# --------------------------------------------------------------------------------------
# IntensityNorm
# --------------------------------------------------------------------------------------


def test_intensity_norm_makes_the_foreground_standard_normal() -> None:
    case = _sphere_case(seed=3)
    fingerprint = collect_fingerprint([case])
    normalized = IntensityNorm(
        mean=fingerprint.foreground_mean, std=fingerprint.foreground_std
    ).apply(case)
    # The evidence rule is the census's: labelled AND foreground (the mask
    # may exclude part of the ball) — select the same voxels the
    # fingerprint averaged over.
    foreground_region = (case.label > 0) & (case.mask[0] > 0)
    foreground = normalized.image[0][foreground_region]
    # The whole point of the corpus-level z-score: after it, the structure
    # the net must learn sits at mean 0 / std 1 wherever the label says so.
    assert float(foreground.mean()) == pytest.approx(0.0, abs=1e-4)
    assert float(foreground.std()) == pytest.approx(1.0, abs=1e-4)
    # Discrete arrays and metadata ride along untouched.
    assert normalized.label is case.label
    assert normalized.mask is case.mask
    assert normalized.spacing_mm == case.spacing_mm
    assert normalized.case_id == case.case_id


def test_intensity_norm_refuses_degenerate_std() -> None:
    case = _sphere_case(seed=4)
    for std in (0.0, 1e-9, -1.0):
        with pytest.raises(ValueError, match="std"):
            IntensityNorm(mean=0.0, std=std).apply(case)


def test_intensity_norm_dict_round_trip() -> None:
    norm = IntensityNorm(mean=187.25, std=412.5)
    assert IntensityNorm.from_dict(norm.to_dict()) == norm


# --------------------------------------------------------------------------------------
# Fingerprint foreground statistics
# --------------------------------------------------------------------------------------


def test_fingerprint_foreground_stats_match_manual_computation() -> None:
    # Three foreground voxels of KNOWN intensity against a -600 background:
    # the census must report exactly their mean and population std.
    image = np.full((1, 8, 8, 8), -600.0, dtype=np.float32)
    label = np.zeros((8, 8, 8), dtype=np.int64)
    mask = np.ones_like(image)
    image[0, 1, 1, 1] = 180.0
    image[0, 2, 2, 2] = 200.0
    image[0, 3, 3, 3] = 220.0
    label[1, 1, 1] = label[2, 2, 2] = label[3, 3, 3] = 1
    fingerprint = collect_fingerprint([Case(image=image, label=label, mask=mask,
                                            spacing_mm=(1.0, 1.0, 1.0))])
    assert fingerprint.foreground_mean == pytest.approx(200.0)
    assert fingerprint.foreground_std == pytest.approx(np.std([180.0, 200.0, 220.0]))


def test_fingerprint_foreground_stats_respect_the_mask() -> None:
    # A foreground-looking voxel the mask does NOT label is not evidence:
    # it must not move the intensity statistics, exactly like the census.
    image = np.full((1, 8, 8, 8), -600.0, dtype=np.float32)
    label = np.zeros((8, 8, 8), dtype=np.int64)
    mask = np.ones_like(image)
    image[0, 1, 1, 1] = 200.0
    image[0, 6, 6, 6] = 9999.0  # fg intensity, but the mask will exclude it
    label[1, 1, 1] = 1
    label[6, 6, 6] = 1
    mask[:, 6, 6, 6] = 0.0
    fingerprint = collect_fingerprint([Case(image=image, label=label, mask=mask,
                                            spacing_mm=(1.0, 1.0, 1.0))])
    assert fingerprint.foreground_mean == pytest.approx(200.0)
    assert fingerprint.foreground_std == pytest.approx(0.0, abs=1e-6)


def test_fingerprint_without_foreground_reports_zero_stats() -> None:
    case = _sphere_case(seed=5)
    background_only = Case(image=case.image, label=np.zeros_like(case.label),
                           mask=case.mask, spacing_mm=case.spacing_mm)
    fingerprint = collect_fingerprint([background_only])
    assert fingerprint.foreground_mean == 0.0
    assert fingerprint.foreground_std == 0.0


# --------------------------------------------------------------------------------------
# resample / resample_case
# --------------------------------------------------------------------------------------


def test_resample_image_downsamples_by_the_spacing_ratio() -> None:
    image = np.random.default_rng(0).normal(size=(1, 32, 32, 32)).astype(np.float32)
    down = resample(image, (1.0, 1.0, 1.0), (2.0, 2.0, 2.0), order=3)
    assert down.shape == (1, 16, 16, 16)
    up = resample(down, (2.0, 2.0, 2.0), (1.0, 1.0, 1.0), order=3)
    assert up.shape == image.shape


def test_resample_image_round_trip_on_a_smooth_field() -> None:
    # Downsampling is a low-pass: the peak of even a smooth field loses a
    # few percent to cubic interpolation, so the honest contract is shape,
    # strong correlation and a small FRACTION of the dynamic range — not
    # bit-exactness, which no resampler could promise.
    grid = np.meshgrid(
        np.linspace(-1.0, 1.0, 32), np.linspace(-1.0, 1.0, 32),
        np.linspace(-1.0, 1.0, 32), indexing="ij",
    )
    image = (500.0 * np.exp(-(grid[0] ** 2 + grid[1] ** 2 + grid[2] ** 2) / 0.5)
             ).astype(np.float32)[None]
    down = resample(image, (1.0, 1.0, 1.0), (2.0, 2.0, 2.0), order=3)
    back = resample(down, (2.0, 2.0, 2.0), (1.0, 1.0, 1.0), order=3)
    span = float(image.max() - image.min())
    assert back.shape == image.shape
    corr = float(np.corrcoef(back.ravel(), image.ravel())[0, 1])
    # 0.995+: structure fully survives; the residual is the low-pass's peak
    # attenuation, which is what downsampling IS.
    assert corr > 0.99, f"round trip scrambled the field: correlation={corr}"
    assert np.allclose(back, image, atol=0.2 * span)


def test_resample_label_preserves_the_label_set() -> None:
    label = np.zeros((30, 30, 30), dtype=np.int64)
    label[4:12, 4:12, 4:12] = 1
    label[16:26, 16:26, 16:26] = 2
    out = resample(label, (1.0, 1.0, 1.0), (1.7, 0.9, 1.3), order=0)
    assert out.dtype == label.dtype
    # Nearest-neighbour may drop a tiny class but NEVER invents one: every
    # output value must be a label the input actually carried.
    assert set(np.unique(out)) <= {0, 1, 2}
    assert {1, 2} <= set(np.unique(out))


def test_resample_case_carries_spacing_id_and_discrete_arrays() -> None:
    case = _sphere_case(seed=6)
    out = resample_case(case, (2.0, 2.0, 2.0))
    assert out.spacing_mm == (2.0, 2.0, 2.0)
    assert out.case_id == case.case_id
    assert out.image.shape == (1, 12, 12, 12)
    assert out.label.shape == (12, 12, 12)
    assert out.mask.shape == (1, 12, 12, 12)
    assert set(np.unique(out.label)) <= {0, 1}
    assert set(np.unique(out.mask)) <= {0.0, 1.0}
    # The discrete contract: only values the input carried may come out.
    assert out.image.dtype == np.float32


# --------------------------------------------------------------------------------------
# The plan decides, and refuses
# --------------------------------------------------------------------------------------


def test_plan_records_normalization_and_target_spacing() -> None:
    cases = [_sphere_case(seed=s) for s in range(4)]
    plan = plan_from_fingerprint(collect_fingerprint(cases), "cpu")
    assert plan.target_spacing == plan.fingerprint.median_spacing
    normalization = plan.normalization
    assert normalization.mean == pytest.approx(plan.fingerprint.foreground_mean)
    assert normalization.std == pytest.approx(plan.fingerprint.foreground_std)
    assert any("normalization" in reason and "resampling target" in reason
               for reason in plan.reasons)


def test_plan_refuses_corpus_without_foreground_intensity_spread() -> None:
    # An all-background corpus: the fingerprint honestly reports zero
    # statistics, and the plan must refuse to normalize it — silently
    # z-scoring by a zero std would fail much further from the cause.
    case = _sphere_case(seed=7)
    background_only = Case(image=case.image, label=np.zeros_like(case.label),
                           mask=case.mask, spacing_mm=case.spacing_mm)
    plan = plan_from_fingerprint(collect_fingerprint([background_only]), "cpu")
    with pytest.raises(ValueError, match="foreground"):
        _ = plan.normalization


def test_preprocess_cases_skips_matching_spacing_and_normalizes() -> None:
    cases = [_sphere_case(seed=s) for s in range(3)]
    fingerprint = collect_fingerprint(cases)
    preprocessing = Preprocessing(
        target_spacing=fingerprint.median_spacing,
        normalization=plan_from_fingerprint(fingerprint, "cpu").normalization,
    )
    out = preprocess_cases(cases, preprocessing)
    # Pooled over the whole corpus the z-score is exact BY CONSTRUCTION (the
    # fingerprint's statistics are linear sums over this very set); a single
    # case's foreground mean may sit off zero by its own sampling error.
    pooled_fg = np.concatenate([
        processed.image[0][(original.label > 0) & (original.mask[0] > 0)]
        for original, processed in zip(cases, out)
    ])
    assert float(pooled_fg.mean()) == pytest.approx(0.0, abs=1e-4)
    assert float(pooled_fg.std()) == pytest.approx(1.0, abs=1e-4)


def test_preprocessing_dict_round_trip_and_identity() -> None:
    preprocessing = Preprocessing(
        target_spacing=(1.25, 0.7, 0.7),
        normalization=IntensityNorm(mean=187.25, std=412.5),
    )
    assert Preprocessing.from_dict(preprocessing.to_dict()) == preprocessing
    identity = Preprocessing.identity()
    assert identity.normalization.normalize(
        np.asarray([[[[-600.0]]]], dtype=np.float32)
    ) == pytest.approx(np.asarray([[[[-600.0]]]], dtype=np.float32))


# --------------------------------------------------------------------------------------
# End to end: two spacings in the corpus, predict on the raw foreign grid
# --------------------------------------------------------------------------------------


def _two_spacing_corpus(tmp_path: Path) -> tuple[Path, Case]:
    """Four cases at 1.0 mm (the median) and two at 2.0 mm, as .npz files —
    plus a held-out RAW 2.0 mm case the fit never sees."""
    out = tmp_path / "cases"
    out.mkdir()
    for seed in range(4):
        case = _sphere_case(seed=seed)
        np.savez(out / f"case-{seed}.npz", image=case.image, label=case.label,
                 mask=case.mask, spacing_mm=np.asarray(case.spacing_mm))
    for seed in (100, 101):
        case = _sphere_case(seed=seed, spacing=(2.0, 2.0, 2.0))
        np.savez(out / f"case-{seed}.npz", image=case.image, label=case.label,
                 mask=case.mask, spacing_mm=np.asarray(case.spacing_mm))
    held_out = _sphere_case(seed=999, spacing=(2.0, 2.0, 2.0))
    return out, held_out


def _foreground_dice(case: Case, label: np.ndarray) -> float:
    supervised = case.mask[0] > 0
    fg = (case.label == 1) & supervised
    denom = float((label == 1).sum() + fg.sum())
    return 2.0 * float((label == 1)[fg].sum()) / denom if denom else 0.0


def test_predict_pins_the_label_to_the_exact_input_grid_on_ugly_ratios() -> None:
    """Zoom rounds its output extent, so a non-integer spacing ratio can
    round-trip one voxel OFF the input shape — the label-grid contract is
    the input's EXACT shape."""
    from medos_trainer.vanilla.infer import SlidingWindowPredictor

    torch.manual_seed(0)
    net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                features=(4, 8, 16), deep_supervision=False))
    predictor = SlidingWindowPredictor(
        net, patch_size=(16, 16, 16),
        preprocessing=Preprocessing(
            target_spacing=(1.0, 1.0, 1.0),
            normalization=IntensityNorm(mean=200.0, std=25.0),
        ),
    )
    image = np.random.default_rng(0).normal(size=(1, 25, 27, 23)).astype(np.float32)
    label, probs = predictor.predict(image, spacing_mm=(0.7, 0.9, 1.13))
    assert label.shape == (25, 27, 23)
    assert probs.shape == (2, 25, 27, 23)


def test_predict_cli_threads_the_case_spacing(tmp_path) -> None:
    """The predict subcommand must read `spacing_mm` out of the case npz
    (load_cases_dir already does; the CLI did not, before preprocessing)
    and pass it through — a foreign-spacing case must come back labelled
    on ITS OWN grid."""
    from medos_trainer.__main__ import main

    bundle = tmp_path / "bundle"
    net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                features=(4, 8, 16), deep_supervision=True))
    preprocessing = Preprocessing(
        target_spacing=(1.0, 1.0, 1.0), normalization=IntensityNorm(200.0, 25.0)
    )
    save_inference_bundle(bundle, net, {"epoch": 0},
                          patch_size=(16, 16, 16), preprocessing=preprocessing)
    case = _sphere_case(seed=21, spacing=(2.0, 2.0, 2.0))
    case_path = tmp_path / "case.npz"
    np.savez(case_path, image=case.image, label=case.label, mask=case.mask,
             spacing_mm=np.asarray(case.spacing_mm))
    out_path = tmp_path / "pred.npz"

    rc = main(["predict", "--checkpoint-dir", str(bundle),
               "--input", str(case_path), "--output", str(out_path)])
    assert rc == 0
    with np.load(out_path) as z:
        assert z["label"].shape == case.label.shape
        assert z["probabilities"].shape == (2,) + case.label.shape


def test_fit_resamples_and_the_bundle_predicts_back_on_the_input_grid(tmp_path) -> None:
    corpus, held_out = _two_spacing_corpus(tmp_path)
    bundle = tmp_path / "bundle"
    # foreground_prob=1.0 and this step count are the toy recipe the suite
    # already trusts (test_vanilla_infer's end-to-end, toy_pipeline.py): on
    # a six-case corpus, one background patch in three would let the net
    # settle in the all-background minimum and the pipeline — not the
    # preprocessing — would be what failed.
    summary = fit_command(corpus, "cpu", bundle, epochs=8, steps_per_epoch=8,
                          foreground_prob=1.0)

    # The decision is recorded twice: in the summary and in the bundle.
    written = json.loads((bundle / "preprocess.json").read_text(encoding="utf-8"))
    assert summary["preprocessing"] == written
    # ... and the record round-trips exactly.
    assert Preprocessing.from_dict(written).to_dict() == written

    predictor = load_predictor(bundle)
    assert predictor.preprocessing is not None
    assert predictor.preprocessing.target_spacing == (1.0, 1.0, 1.0)

    # The RAW 2.0 mm case, predicted with its OWN spacing named: the bundle
    # resamples up to the 1.0 mm training grid and BACK, so the label lives
    # on the input grid — the shape is the contract.
    label, probs = predictor.predict(held_out.image, spacing_mm=held_out.spacing_mm)
    assert label.shape == held_out.label.shape
    assert probs.shape == (2,) + held_out.label.shape
    learned = _foreground_dice(held_out, label)

    # The baseline: the same architecture never trained, but handed the same
    # preprocessing — only the weights differ.
    untrained_dir = tmp_path / "untrained"
    torch.manual_seed(1234)
    fresh = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                  features=(16, 32, 64, 128), deep_supervision=True))
    save_inference_bundle(untrained_dir, fresh, {"epoch": -1},
                          patch_size=tuple(predictor.patch_size),
                          preprocessing=predictor.preprocessing)
    naive_label, _ = load_predictor(untrained_dir).predict(
        held_out.image, spacing_mm=held_out.spacing_mm
    )
    naive = _foreground_dice(held_out, naive_label)

    # Learning happened on the resampled corpus, and it transfers across
    # spacings — that is the whole benchmark lesson.
    assert learned > 0.2, f"predictor did not learn the toy: dice={learned}"
    assert learned > naive, f"no improvement over untrained: {learned} vs {naive}"
