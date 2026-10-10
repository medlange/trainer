# SPDX-License-Identifier: Apache-2.0
"""Checkpoint selection: what chooses the best model, and what survives.

THE BENCHMARK LESSON UNDER TEST (docs/benchmark-pulmo-2026-10-07.md,
"Long-run continuation and the checkpoint-selection lesson", 2026-10-10):
the 250-epoch PulmoAI run kept improving the masked-patch val proxy
(0.3261 -> 0.3138 at epoch 121), yet the epoch-121 bundle that proxy
selected scored 0.438 fg Dice full-volume — and had OVERWRITTEN the
epoch-111 bundle's 0.539. So: (1) the selector must score what deployment
scores — full-volume validation foreground Dice, "volume_dice" — and
(2) the best model must never be a single overwriteable file — top-K
snapshots under checkpoints/.

THE ADVERSARIAL MECHANICS, shared by the tests below: selection is driven
by monkeypatching `VanillaTrainer.volume_selection_score` to a
DETERMINISTIC SEQUENCE of per-epoch scores. Training itself runs for real
on the toy corpus; only the metric is scripted. A sequence that RISES to a
peak mid-run and then COLLAPSES reproduces the benchmark's failure shape —
the run's LAST epoch is the one a last-epoch (or single-overwrite)
policy would keep, and it is exactly the wrong one — and the assertions
pin that the live bundle, the top-K index and the on-disk snapshots all
still point at the PEAK epoch.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from medos_trainer.standalone import fit_command
from medos_trainer.vanilla.infer import (
    SlidingWindowPredictor,
    load_predictor,
    served_net,
)
from medos_trainer.vanilla.nets import UNetConfig, build_unet
from medos_trainer.vanilla.trainer import FitPlan, VanillaTrainer
from test_vanilla_data import _toy_cases


def _write_cases(tmp_path, n: int = 5, shape=(20, 20, 20)):
    """Random unlearnable-ish cases through the planned door (fit_command),
    same shape as the resume tests' fixture."""
    rng = np.random.default_rng(0)
    out = tmp_path / "cases"
    out.mkdir()
    for i in range(n):
        image = rng.normal(-600.0, 50.0, (1, *shape)).astype(np.float32)
        label = (rng.random(shape) < 0.05).astype(np.int64)
        np.savez(out / f"case-{i}.npz", image=image, label=label,
                 spacing_mm=np.asarray((1.0, 1.0, 1.0)))
    return out


def _toy_trainer(selection: str = "volume_dice", keep_checkpoints: int = 3,
                 epochs: int = 5) -> VanillaTrainer:
    torch.manual_seed(0)
    net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                features=(4, 8, 16), deep_supervision=True))
    plan = FitPlan(patch_size=(16, 16, 16), batch_size=2, steps_per_epoch=2,
                   epochs=epochs, foreground_prob=1.0, selection=selection,
                   keep_checkpoints=keep_checkpoints)
    return VanillaTrainer(net, num_classes=2, plan=plan)


def _scripted_selection(monkeypatch, scores: list[float]) -> None:
    """Replace the volume selector with a deterministic per-epoch sequence
    (one entry per epoch, consumed in order)."""
    remaining = iter(scores)

    def fake(self, val_cases, preprocessing=None):
        return next(remaining)

    monkeypatch.setattr(VanillaTrainer, "volume_selection_score", fake)


def _index(bundle) -> list[dict]:
    return json.loads(
        (bundle / "checkpoints" / "index.json").read_text(encoding="utf-8")
    )


def test_fitplan_selection_choices_and_counts_are_validated() -> None:
    with pytest.raises(ValueError, match="selection"):
        FitPlan(patch_size=(16, 16, 16), selection="dice")
    with pytest.raises(ValueError, match="selection"):
        FitPlan(patch_size=(16, 16, 16), selection="")
    with pytest.raises(ValueError, match="selection_cases"):
        FitPlan(patch_size=(16, 16, 16), selection_cases=0)
    with pytest.raises(ValueError, match="selection_cases"):
        FitPlan(patch_size=(16, 16, 16), selection_cases=-2)
    with pytest.raises(ValueError, match="keep_checkpoints"):
        FitPlan(patch_size=(16, 16, 16), keep_checkpoints=0)
    # The defaults: the historical selector and top-K retention.
    plan = FitPlan(patch_size=(16, 16, 16))
    assert plan.selection == "patch_dice"
    assert plan.selection_cases == 4
    assert plan.keep_checkpoints == 3


def test_patch_keep_one_is_the_historical_layout(tmp_path, monkeypatch) -> None:
    """keep_checkpoints=1: exactly the pre-top-K file set — the live bundle
    and its resume record, NO checkpoints/ directory."""
    vals = iter([0.40, 0.30, 0.20])
    monkeypatch.setattr(
        VanillaTrainer, "validate",
        lambda self, cases, rng, patches_per_case=2: next(vals),
    )
    trainer = _toy_trainer(selection="patch_dice", keep_checkpoints=1,
                           epochs=3)
    train, val = _toy_cases(4, seed=0), _toy_cases(2, seed=100)
    result = trainer.fit(train, val, np.random.default_rng(0),
                         out_dir=tmp_path)

    assert (tmp_path / "model.pt").is_file()
    assert (tmp_path / "checkpoint.json").is_file()
    assert (tmp_path / "training_state.pt").is_file()
    assert not (tmp_path / "checkpoints").exists()
    # The historical contract, byte-identical: the record carries the loss,
    # the result the min loss.
    record = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert record["val_masked_dice_loss"] == pytest.approx(0.20)
    assert record["selection_score"] == pytest.approx(-0.20)
    assert {"epoch", "loss", "val_masked_dice_loss", "lr"} <= set(record)
    assert result["best_val_masked_dice_loss"] == pytest.approx(0.20)


def test_patch_keep_two_writes_topk_snapshots(tmp_path, monkeypatch) -> None:
    """keep_checkpoints>=2 on the patch selector: every improving epoch
    snapshots, the index sorts by score (negated loss) descending."""
    vals = iter([0.40, 0.30, 0.20])
    monkeypatch.setattr(
        VanillaTrainer, "validate",
        lambda self, cases, rng, patches_per_case=2: next(vals),
    )
    trainer = _toy_trainer(selection="patch_dice", keep_checkpoints=2,
                           epochs=3)
    train, val = _toy_cases(4, seed=0), _toy_cases(2, seed=100)
    trainer.fit(train, val, np.random.default_rng(0), out_dir=tmp_path)

    entries = _index(tmp_path)
    assert [e["epoch"] for e in entries] == [2, 1]  # best first
    assert [e["score"] for e in entries] == pytest.approx([-0.20, -0.30])
    for e in entries:
        assert (tmp_path / "checkpoints" / e["dir"] / "model.pt").is_file()
        assert (tmp_path / "checkpoints" / e["dir"] / "net_config.json").is_file()
    assert not (tmp_path / "checkpoints" / "epoch-0").exists()


def test_volume_selection_survives_a_lying_final_epoch(tmp_path,
                                                       monkeypatch) -> None:
    """THE BENCHMARK FAILURE, REPRODUCED ON THE TOY. The scripted selection
    peaks at epoch 2 (0.90) and the final epoch collapses to 0.10 — the
    proxy-improves-while-deployment-collapses shape that turned 0.539 into
    0.438. The live bundle, the index head and the peak snapshot must ALL
    be epoch 2, and the peak snapshot must predict identically to the live
    bundle (it IS the same weights)."""
    _scripted_selection(monkeypatch, [0.30, 0.50, 0.90, 0.40, 0.10])
    trainer = _toy_trainer(keep_checkpoints=3, epochs=5)
    train, val = _toy_cases(4, seed=0), _toy_cases(2, seed=100)
    result = trainer.fit(train, val, np.random.default_rng(0),
                         out_dir=tmp_path)

    record = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert record["epoch"] == 2
    assert record["selection_score"] == pytest.approx(0.90)
    assert record["volume_dice"] == pytest.approx(0.90)
    assert "val_masked_dice_loss" not in record
    # The live bundle mirrors the index head.
    entries = _index(tmp_path)
    assert entries[0]["epoch"] == 2
    assert entries[0]["score"] == pytest.approx(0.90)
    assert [e["score"] for e in entries] == sorted(
        (e["score"] for e in entries), reverse=True
    )
    assert result["best_selection_score"] == pytest.approx(0.90)
    assert result["best_val_masked_dice_loss"] is None

    # The peak snapshot predicts IDENTICALLY to the live bundle — the best
    # model on disk is the best model, whichever door loads it.
    peak_dir = tmp_path / "checkpoints" / entries[0]["dir"]
    case = _toy_cases(1, seed=7)[0]
    live_label, live_probs = load_predictor(tmp_path).predict(case.image)
    peak_label, peak_probs = load_predictor(peak_dir).predict(case.image)
    assert np.array_equal(live_label, peak_label)
    assert np.array_equal(live_probs, peak_probs)


def test_topk_eviction_keeps_only_the_best_snapshots(tmp_path,
                                                     monkeypatch) -> None:
    """A rising-then-falling score sequence with K=2: three epochs improved
    (snapshots written), the worst is evicted from DISK — only the two best
    snapshots remain, and the index is the eviction order."""
    _scripted_selection(monkeypatch, [0.10, 0.50, 0.90, 0.40, 0.20])
    trainer = _toy_trainer(keep_checkpoints=2, epochs=5)
    train, val = _toy_cases(4, seed=0), _toy_cases(2, seed=100)
    trainer.fit(train, val, np.random.default_rng(0), out_dir=tmp_path)

    entries = _index(tmp_path)
    assert [(e["epoch"], e["score"]) for e in entries] == [
        (2, pytest.approx(0.90)), (1, pytest.approx(0.50)),
    ]
    on_disk = sorted(
        d.name for d in (tmp_path / "checkpoints").iterdir()
        if d.is_dir() and d.name.startswith("epoch-")
    )
    assert set(on_disk) == {e["dir"] for e in entries}
    assert len(on_disk) == 2


def test_planned_run_selects_volume_dice() -> None:
    """A real plan carries the benchmark's fix; hand-written plans default
    to the historical proxy (asserted in test_fitplan_selection_...)."""
    from medos_trainer.vanilla.plan import (
        collect_fingerprint,
        plan_from_fingerprint,
    )

    plan = plan_from_fingerprint(collect_fingerprint(_toy_cases(4, seed=0)),
                                 "cpu")
    fit_plan = plan.fit_plan()
    assert fit_plan.selection == "volume_dice"
    assert fit_plan.keep_checkpoints >= 2
    assert fit_plan.selection_cases >= 1


def test_volume_dice_matches_the_evaluators_definition() -> None:
    """The selector's number IS the deployment number: same net, same cases,
    same sliding window — volume_dice must reproduce _dice_rows' aggregate
    foreground mean exactly (the training net's DS tuple included)."""
    from medos_trainer.standalone import _dice_rows
    from medos_trainer.vanilla.selection import volume_dice

    torch.manual_seed(0)
    net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                features=(4, 8, 16), deep_supervision=True))
    cases = _toy_cases(2, seed=5)
    got = volume_dice(net, cases, "cpu", max_cases=2, patch_size=(16, 16, 16))
    assert 0.0 <= got <= 1.0

    # _dice_rows over the served twin is the evaluator's own code path; the
    # selector must agree with it to the last decimal.
    predictor = SlidingWindowPredictor(
        served_net(net), patch_size=(16, 16, 16), device="cpu"
    )
    report = _dice_rows(predictor, cases, max_cases=2, num_classes=2)
    assert got == pytest.approx(report["aggregate"]["foreground_mean_mean"],
                                abs=1e-6)

    # max_cases caps which cases are scored, like the evaluator's cap.
    first_only = volume_dice(net, cases, "cpu", max_cases=1,
                             patch_size=(16, 16, 16))
    one = _dice_rows(predictor, cases[:1], max_cases=1, num_classes=2)
    assert first_only == pytest.approx(
        one["aggregate"]["foreground_mean_mean"], abs=1e-6
    )


def test_resume_restores_the_index_head_as_the_bar(tmp_path, monkeypatch) -> None:
    """The run-wide best survives a resume: the first run peaks at epoch 0;
    the resumed run's scores improve on its own last epoch but never beat
    the index head — so the artifact must STILL be epoch 0, and eviction
    must not have touched the peak snapshot."""
    cases = _write_cases(tmp_path)
    bundle = tmp_path / "bundle"

    _scripted_selection(monkeypatch, [0.90, 0.10, 0.05, 0.20, 0.30, 0.15, 0.25])
    first = fit_command(cases, "cpu", bundle, epochs=3, steps_per_epoch=1,
                        seed=0)
    assert first["best_selection_score"] == pytest.approx(0.90)
    record = json.loads((bundle / "checkpoint.json").read_text(encoding="utf-8"))
    assert record["epoch"] == 0
    assert (bundle / "checkpoints" / "epoch-0" / "model.pt").is_file()

    # No monkeypatch release: the SAME scripted sequence continues to flow.
    # A resume continues from the BEST checkpoint's epoch (0 + 1), and every
    # one of its scores improves on the resumed stream's start but never on
    # the index head — no snapshot may be written and the artifact must not
    # move.
    second = fit_command(cases, "cpu", bundle, epochs=5, steps_per_epoch=1,
                         seed=0, resume_from=bundle)
    record = json.loads((bundle / "checkpoint.json").read_text(encoding="utf-8"))
    assert record["epoch"] == 0, "the resumed run regressed the artifact"
    assert record["selection_score"] == pytest.approx(0.90)
    assert second["best_selection_score"] == pytest.approx(0.90)
    assert [r["epoch"] for r in second["history"]] == [1, 2, 3, 4]
    # The index is unchanged — still just the peak — and its snapshot is
    # intact on disk.
    entries = _index(bundle)
    assert [(e["epoch"], e["score"]) for e in entries] == [(0, pytest.approx(0.90))]
    assert (bundle / "checkpoints" / "epoch-0" / "model.pt").is_file()


def test_deployment_dice_equals_the_evaluator_aggregate(tmp_path) -> None:
    """The only non-lying selector must BE the evaluator: `deployment_dice`
    on a net must equal `evaluate_predictor`'s aggregate foreground Dice on
    a predictor built from the SAME net over the SAME raw cases. The
    benchmark caught the training-grid scorer ranking epochs by grid-overfit
    (selection 188>172>188-ranked-wrong vs deployment 115>172>188); this pin
    makes the selector's grid the evaluator's grid, by construction."""
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).parent))
    from medos_trainer.standalone import evaluate_predictor
    from medos_trainer.vanilla.infer import load_predictor, save_inference_bundle
    from medos_trainer.vanilla.nets import UNetConfig, build_unet
    from medos_trainer.vanilla.preprocess import Preprocessing
    from medos_trainer.vanilla.selection import deployment_dice
    from test_vanilla_data import _toy_cases

    net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                features=(4, 8, 16), deep_supervision=False))
    cases = _toy_cases(2, seed=5)
    preprocessing = Preprocessing.identity()

    save_inference_bundle(tmp_path, net, {"epoch": 0, "loss": 0.0,
                                          "val_masked_dice_loss": 0.0,
                                          "lr": 0.0}, patch_size=(16, 16, 16))
    (tmp_path / "preprocess.json").write_text(
        '{"target_spacing": [1.0, 1.0, 1.0], '
        '"normalization": {"mean": 0.0, "std": 1.0}}', encoding="utf-8")
    rep = evaluate_predictor(load_predictor(tmp_path), cases, out=None)
    expected = rep["aggregate"]["foreground_mean_mean"]
    score = deployment_dice(net, preprocessing, (16, 16, 16), cases,
                            "cpu", max_cases=2)
    assert score == pytest.approx(expected, abs=1e-9), (score, expected)
