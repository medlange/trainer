# SPDX-License-Identifier: Apache-2.0
"""Fold ensembles: probability averaging must not collapse.

THE PROPERTIES UNDER TEST: an `EnsemblePredictor` composes sliding-window
members (each keeping its own patch size/overlap/device), refuses class-table
mismatches naming the offender, and — on a toy a single model already learns —
scores at least the WORSE member and no less than 90% of the BETTER one.
Averaging can in principle dip below a member on a finite set; the claim
being pinned is that on data like this it does not, which is the property a
user means by "the ensemble should not collapse".
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from medos_trainer.__main__ import main
from medos_trainer.standalone import _dice_rows, crossval_command
from medos_trainer.vanilla.infer import (
    EnsemblePredictor,
    load_ensemble,
    load_predictor,
    save_inference_bundle,
)
from medos_trainer.vanilla.nets import UNetConfig, build_unet
from medos_trainer.vanilla.trainer import FitPlan, VanillaTrainer


def _member_bundle(tmp_path, seed: int, name: str):
    """One toy bundle trained from `seed` — an ensemble member with real,
    seed-dependent weights (the whole point is that the members disagree)."""
    from test_vanilla_data import _toy_cases

    torch.manual_seed(seed)
    net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                features=(4, 8, 16), deep_supervision=True))
    plan = FitPlan(patch_size=(16, 16, 16), batch_size=2, steps_per_epoch=8,
                   epochs=8, foreground_prob=1.0)
    trainer = VanillaTrainer(net, num_classes=2, plan=plan)
    train, val = _toy_cases(4, seed=0), _toy_cases(2, seed=100)
    trainer.fit(train, val, np.random.default_rng(seed),
                out_dir=tmp_path / name)
    return tmp_path / name


def _two_members(tmp_path):
    dirs = [_member_bundle(tmp_path, 0, "m0"), _member_bundle(tmp_path, 1, "m1")]
    return dirs, [load_predictor(d) for d in dirs]


def test_ensemble_predict_agrees_with_member_shapes(tmp_path) -> None:
    from test_vanilla_data import _toy_cases

    _, members = _two_members(tmp_path)
    ensemble = EnsemblePredictor(members)
    assert len(ensemble.members) == 2
    case = _toy_cases(1, seed=7)[0]
    label, probs = ensemble.predict(case.image)
    for member in members:
        m_label, m_probs = member.predict(case.image)
        assert label.shape == m_label.shape == case.label.shape
        assert probs.shape == m_probs.shape == (2,) + case.label.shape
    # probabilities are a mean: bounded by the members' range at every voxel
    member_probs = np.stack([m.predict(case.image)[1] for m in members])
    assert np.all(probs <= member_probs.max(axis=0) + 1e-6)
    assert np.all(probs >= member_probs.min(axis=0) - 1e-6)


def test_ensemble_dice_stands_between_the_members(tmp_path) -> None:
    from test_vanilla_data import _toy_cases

    _, members = _two_members(tmp_path)
    ensemble = EnsemblePredictor(members)
    val = _toy_cases(3, seed=100)

    member_fg = [
        _dice_rows(m, val)["aggregate"]["foreground_mean_mean"] for m in members
    ]
    ensemble_fg = _dice_rows(ensemble, val, num_classes=ensemble.num_classes)[
        "aggregate"
    ]["foreground_mean_mean"]
    worse, better = min(member_fg), max(member_fg)
    assert ensemble_fg >= worse, (
        f"ensemble {ensemble_fg:.4f} below its worse member {worse:.4f}"
    )
    assert ensemble_fg >= 0.9 * better, (
        f"ensemble {ensemble_fg:.4f} collapsed: better member {better:.4f}"
    )


def test_ensemble_refuses_class_mismatch_naming_the_offender(tmp_path) -> None:
    odd = build_unet(UNetConfig(input_channels=1, num_classes=3,
                                features=(4, 8, 16), deep_supervision=True))
    save_inference_bundle(tmp_path / "odd", odd, {"epoch": 0}, patch_size=(16, 16, 16))
    even = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                 features=(4, 8, 16), deep_supervision=True))
    save_inference_bundle(tmp_path / "even", even, {"epoch": 0},
                          patch_size=(16, 16, 16))
    member_a = load_predictor(tmp_path / "even")
    member_b = load_predictor(tmp_path / "odd")
    with pytest.raises(ValueError, match="member 1.*num_classes=3"):
        EnsemblePredictor([member_a, member_b])
    with pytest.raises(ValueError, match="at least one member"):
        EnsemblePredictor([])


def test_load_ensemble_composes_one_predictor_per_dir(tmp_path) -> None:
    dirs, members = _two_members(tmp_path)
    ensemble = load_ensemble(dirs)
    assert len(ensemble.members) == len(members)
    assert ensemble.num_classes == 2


def test_predict_cli_ensemble_dir_smoke(tmp_path) -> None:
    """A tiny real cross-validation output drives the ensemble door end to
    end: both fold bundles vote, the report's aggregate is printed, and the
    output npz carries the agreed shapes."""
    from test_vanilla_data import _toy_cases

    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    for i, case in enumerate(_toy_cases(4, seed=0)):
        np.savez(cases_dir / f"case-{i}.npz", image=case.image, label=case.label,
                 mask=case.mask)
    cv_dir = tmp_path / "cv"
    rc = main(["vanilla-crossval", "--data", str(cases_dir), "--preset", "cpu",
               "--out", str(cv_dir), "--folds", "2", "--epochs", "1",
               "--steps-per-epoch", "1"])
    assert rc == 0
    report = json.loads((cv_dir / "report.json").read_text(encoding="utf-8"))
    assert report["ensemble"]["cases_used"] == 4
    assert "per_class_dice_mean" in report["ensemble"]["aggregate"]

    case = _toy_cases(1, seed=11)[0]
    case_path = tmp_path / "one.npz"
    np.savez(case_path, image=case.image, label=case.label, mask=case.mask)
    out_path = tmp_path / "pred.npz"
    rc = main(["predict", "--ensemble-dir", str(cv_dir),
               "--input", str(case_path), "--output", str(out_path)])
    assert rc == 0
    with np.load(out_path) as z:
        assert z["label"].shape == case.label.shape
        assert z["probabilities"].shape == (2,) + case.label.shape


def test_predict_cli_ensemble_dir_refuses_an_empty_crossval_output(tmp_path) -> None:
    empty = tmp_path / "cv"
    empty.mkdir()
    case_path = tmp_path / "one.npz"
    np.savez(case_path, image=np.zeros((1, 8, 8, 8), dtype=np.float32),
             label=np.zeros((8, 8, 8), dtype=np.int64))
    rc = main(["predict", "--ensemble-dir", str(empty),
               "--input", str(case_path), "--output", str(tmp_path / "o.npz")])
    assert rc == 2


def test_crossval_report_carries_the_ensemble_aggregate(tmp_path) -> None:
    """The ensemble row is computed by the SAME `_dice_rows` as everything
    else, over the union of the folds' val cases — structurally: the row
    exists, names every case once, and the aggregate matches a from-scratch
    ensemble evaluation over the same union."""
    from test_vanilla_data import _toy_cases

    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    for i, case in enumerate(_toy_cases(4, seed=0)):
        np.savez(cases_dir / f"case-{i}.npz", image=case.image, label=case.label,
                 mask=case.mask)
    cv_dir = tmp_path / "cv"
    report = crossval_command(cases_dir, "cpu", cv_dir, folds=2, epochs=1,
                              steps_per_epoch=1)
    ensemble = report["ensemble"]
    assert ensemble["cases_used"] == 4
    assert ensemble["aggregate"]["cases_used"] == 4

    from medos_trainer.standalone import load_cases_dir

    cases = load_cases_dir(cases_dir)
    again = _dice_rows(
        load_ensemble([cv_dir / "fold-0", cv_dir / "fold-1"]), cases
    )["aggregate"]
    assert ensemble["aggregate"]["foreground_mean_mean"] == pytest.approx(
        again["foreground_mean_mean"], abs=1e-9)
