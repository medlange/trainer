# SPDX-License-Identifier: Apache-2.0
"""Post-fit evaluation: per-case Dice, restricted to supervised voxels.

The number is judged two ways: structurally (the report's shape — per-class
lists, aggregate means, case counts) and numerically (the aggregate equals
the mean of the per-case rows, and the per-case rows equal Dice recomputed
by hand from the predictor's own output).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from medos_trainer.__main__ import main
from medos_trainer.standalone import evaluate_command
from medos_trainer.vanilla.infer import load_predictor
from medos_trainer.vanilla.nets import UNetConfig, build_unet
from medos_trainer.vanilla.trainer import FitPlan, VanillaTrainer


def _trained_bundle(tmp_path):
    from test_vanilla_data import _toy_cases

    torch.manual_seed(0)
    net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                features=(4, 8, 16), deep_supervision=True))
    plan = FitPlan(patch_size=(16, 16, 16), batch_size=2, steps_per_epoch=4,
                   epochs=4, foreground_prob=1.0)
    trainer = VanillaTrainer(net, num_classes=2, plan=plan)
    train, val = _toy_cases(4, seed=0), _toy_cases(2, seed=100)
    trainer.fit(train, val, np.random.default_rng(0), out_dir=tmp_path / "bundle")
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    np.savez(cases_dir / "val-0.npz",
             image=val[0].image, label=val[0].label, mask=val[0].mask)
    np.savez(cases_dir / "val-1.npz",
             image=val[1].image, label=val[1].label, mask=val[1].mask)
    return tmp_path / "bundle", cases_dir, val


def _expected_dice(predictor, case) -> list[float]:
    label, _ = predictor.predict(case.image)
    supervised = case.mask.max(axis=0) > 0
    per_class = []
    for cls in range(predictor.net.config.num_classes):
        predicted = (label == cls) & supervised
        truth = (case.label == cls) & supervised
        denom = int(predicted.sum()) + int(truth.sum())
        per_class.append(2.0 * float((predicted & truth).sum()) / denom if denom else 0.0)
    return per_class


def test_report_shape_and_aggregate_match_hand_computation(tmp_path) -> None:
    bundle, cases_dir, val = _trained_bundle(tmp_path)
    out = tmp_path / "eval.json"
    report = evaluate_command(bundle, cases_dir, out)
    assert json.loads(out.read_text(encoding="utf-8")) == report

    predictor = load_predictor(bundle)
    assert len(report["cases"]) == 2
    for i, (row, case) in enumerate(zip(report["cases"], val)):
        assert row["case_id"] == f"val-{i}"  # the npz filename, not the toy's
        assert len(row["per_class_dice"]) == predictor.net.config.num_classes
        assert row["per_class_dice"] == pytest.approx(_expected_dice(predictor, case))

    manual_class_means = [
        float(np.mean([row["per_class_dice"][cls] for row in report["cases"]]))
        for cls in range(predictor.net.config.num_classes)
    ]
    assert report["aggregate"]["per_class_dice_mean"] == pytest.approx(
        manual_class_means, abs=1e-9)
    assert report["aggregate"]["foreground_mean_mean"] == pytest.approx(
        float(np.mean([row["foreground_mean"] for row in report["cases"]])), abs=1e-9)
    assert report["aggregate"]["cases_used"] == 2


def test_max_cases_caps_the_report(tmp_path) -> None:
    bundle, cases_dir, _ = _trained_bundle(tmp_path)
    report = evaluate_command(bundle, cases_dir, tmp_path / "eval.json", max_cases=1)
    assert len(report["cases"]) == 1
    assert report["aggregate"]["cases_used"] == 1


def test_evaluate_cli_smoke(tmp_path) -> None:
    bundle, cases_dir, _ = _trained_bundle(tmp_path)
    out = tmp_path / "eval.json"
    rc = main(["vanilla-evaluate", "--checkpoint-dir", str(bundle),
               "--data", str(cases_dir), "--out", str(out)])
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["aggregate"]["cases_used"] == 2
