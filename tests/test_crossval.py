# SPDX-License-Identifier: Apache-2.0
"""k-fold cross-validation over a cases directory.

THE PROPERTIES UNDER TEST: folds are deterministic (the assignment is part
of the report and must be reproducible from the seed alone), every case is
validated exactly once, each fold leaves a full bundle, and the aggregate is
the plain statistics of the per-fold scores — nothing smoothed, nothing
hidden.
"""

from __future__ import annotations

import json
import statistics

import numpy as np
import pytest
from medos_trainer.__main__ import main
from medos_trainer.standalone import crossval_command, load_cases_dir


def _write_cases(tmp_path, n: int = 6, shape=(20, 20, 20)):
    rng = np.random.default_rng(0)
    out = tmp_path / "cases"
    out.mkdir()
    for i in range(n):
        image = rng.normal(-600.0, 50.0, (1, *shape)).astype(np.float32)
        label = (rng.random(shape) < 0.05).astype(np.int64)
        np.savez(out / f"case-{i}.npz", image=image, label=label,
                 spacing_mm=np.asarray((1.0, 1.0, 1.0)))
    return out


def test_crossval_writes_bundles_report_and_covers_every_case(tmp_path) -> None:
    cases_dir = _write_cases(tmp_path)
    out = tmp_path / "cv"
    report = crossval_command(cases_dir, "cpu", out, folds=3, epochs=1, steps_per_epoch=1)

    for k in range(3):
        assert (out / f"fold-{k}" / "model.pt").is_file()
        assert (out / f"fold-{k}" / "net_config.json").is_file()
    loaded = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert loaded == report

    assert report["folds"] == 3
    assignment = report["fold_assignment"]
    cases = load_cases_dir(cases_dir)
    assert sorted(assignment) == sorted(c.case_id for c in cases)
    assert sorted(assignment.values()) == [0, 0, 1, 1, 2, 2]

    per_fold = [set(row["val_cases"]) for row in report["results"]]
    union = per_fold[0] | per_fold[1] | per_fold[2]
    assert union == {c.case_id for c in cases}
    for a in range(3):
        for b in range(a + 1, 3):
            assert not (per_fold[a] & per_fold[b]), "folds must not share a case"

    scores = [row["best_val_masked_dice_loss"] for row in report["results"]]
    assert report["aggregate"]["mean"] == pytest.approx(statistics.mean(scores), abs=1e-9)


def test_fold_assignment_is_the_documented_seeded_scheme(tmp_path) -> None:
    """Pin the scheme, not just its determinism: sorted-by-case_id order,
    one default_rng(seed) permutation, rank mod folds."""
    cases_dir = _write_cases(tmp_path)
    report = crossval_command(cases_dir, "cpu", tmp_path / "cv", folds=3,
                              epochs=1, steps_per_epoch=1, seed=7)
    cases = load_cases_dir(cases_dir)
    permutation = np.random.default_rng(7).permutation(len(cases))
    expected = {cases[idx].case_id: int(rank % 3)
                for rank, idx in enumerate(permutation)}
    assert report["fold_assignment"] == expected


def test_crossval_refuses_degenerate_folds(tmp_path) -> None:
    cases_dir = _write_cases(tmp_path)
    with pytest.raises(ValueError, match="at least 2 folds"):
        crossval_command(cases_dir, "cpu", tmp_path / "cv", folds=1)
    with pytest.raises(ValueError, match="empty fold"):
        crossval_command(cases_dir, "cpu", tmp_path / "cv", folds=7)


def test_crossval_cli_smoke(tmp_path) -> None:
    cases_dir = _write_cases(tmp_path)
    out = tmp_path / "cv"
    rc = main(["vanilla-crossval", "--data", str(cases_dir), "--preset", "cpu",
               "--out", str(out), "--folds", "3", "--epochs", "1",
               "--steps-per-epoch", "1"])
    assert rc == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["folds"] == 3
    assert len(report["results"]) == 3
