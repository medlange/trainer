# SPDX-License-Identifier: Apache-2.0
"""Resume: the bundle's training_state.pt must let a run continue honestly.

TWO PROPERTIES, because "resume" is two claims: (1) the OPTIMIZER and
SCHEDULER state come back — a resumed run continues the learning rate the
plateau left behind, not the plan's opening value; (2) the EPOCH NUMBERING
and checkpoint selection stay run-wide — history is the fresh epochs plus
the resumed ones, and the artifact can only improve on the fresh run's best.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from medos_trainer.__main__ import main
from medos_trainer.standalone import fit_command
from medos_trainer.vanilla.nets import UNetConfig, build_unet
from medos_trainer.vanilla.trainer import FitPlan, VanillaTrainer


def _write_cases(tmp_path, n: int = 5, shape=(20, 20, 20)):
    rng = np.random.default_rng(0)
    out = tmp_path / "cases"
    out.mkdir()
    for i in range(n):
        image = rng.normal(-600.0, 50.0, (1, *shape)).astype(np.float32)
        label = (rng.random(shape) < 0.05).astype(np.int64)
        np.savez(out / f"case-{i}.npz", image=image, label=label,
                 spacing_mm=np.asarray((1.0, 1.0, 1.0)))
    return out


def test_save_and_load_state_restores_optimizer_and_scheduler(tmp_path) -> None:
    from test_vanilla_data import _toy_cases

    torch.manual_seed(0)
    net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                features=(4, 8, 16), deep_supervision=True))
    plan = FitPlan(patch_size=(16, 16, 16), batch_size=2, steps_per_epoch=2,
                   epochs=2, foreground_prob=1.0)
    trainer = VanillaTrainer(net, num_classes=2, plan=plan)
    train, val = _toy_cases(4, seed=0), _toy_cases(2, seed=100)
    trainer.fit(train, val, np.random.default_rng(0), out_dir=tmp_path / "bundle")

    # Stall the plateau deterministically so the scheduled lr differs from
    # the plan's opening lr — the assertion below then proves restoration
    # rather than coincidence with the default.
    for _ in range(5):
        trainer.scheduler.step(0.0)
    forced_lr = trainer.optimizer.param_groups[0]["lr"]
    assert forced_lr != plan.learning_rate
    state_path = trainer.save_state(tmp_path / "bundle", epoch=1)

    fresh = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                  features=(4, 8, 16), deep_supervision=True))
    resumed = VanillaTrainer(fresh, num_classes=2, plan=plan)
    state = resumed.load_state(tmp_path / "bundle")
    assert state_path.is_file()
    assert state["epoch"] == 1
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(forced_lr)
    # The net itself came back: same weights as the run that saved them.
    for (n_a, p_a), (n_b, p_b) in zip(
        trainer.net.named_parameters(), resumed.net.named_parameters()
    ):
        assert n_a == n_b
        assert torch.equal(p_a, p_b)


def test_fit_command_resume_continues_epoch_numbering_and_best(tmp_path,
                                                               monkeypatch) -> None:
    """With validation (a LOSS — lower is better) driven deterministically
    downward, the resumed run's checkpoint must be its last epoch and history
    must hold all four."""
    cases = _write_cases(tmp_path)
    bundle = tmp_path / "bundle"

    vals = iter([0.40, 0.30, 0.20, 0.10])

    def fake_validate(self, cases, rng, patches_per_case=2):
        return next(vals)

    monkeypatch.setattr(VanillaTrainer, "validate", fake_validate)

    first = fit_command(cases, "cpu", bundle, epochs=2, steps_per_epoch=1, seed=0)
    assert (bundle / "training_state.pt").is_file()
    assert len(first["history"]) == 2
    assert [r["epoch"] for r in first["history"]] == [0, 1]
    # The scheduler watched a strictly improving loss: lr is untouched.
    assert first["history"][-1]["lr"] == pytest.approx(first["history"][0]["lr"])

    second = fit_command(cases, "cpu", bundle, epochs=4, steps_per_epoch=1,
                         seed=0, resume_from=bundle)
    assert [r["epoch"] for r in second["history"]] == [2, 3]
    record = json.loads((bundle / "checkpoint.json").read_text(encoding="utf-8"))
    assert record["epoch"] == 3
    assert record["val_masked_dice_loss"] == pytest.approx(0.10)
    # The run-wide best is the checkpoint's, and the summary agrees.
    assert second["best_val_masked_dice_loss"] == pytest.approx(0.10)


def test_resume_cli_smoke(tmp_path, monkeypatch) -> None:
    cases = _write_cases(tmp_path)
    bundle = tmp_path / "bundle"
    vals = iter([0.4, 0.3, 0.2, 0.1])
    monkeypatch.setattr(
        VanillaTrainer, "validate",
        lambda self, c, rng, patches_per_case=2: next(vals),
    )
    rc = main(["vanilla-fit", "--data", str(cases), "--preset", "cpu",
               "--out", str(bundle), "--epochs", "2", "--steps-per-epoch", "1"])
    assert rc == 0
    rc = main(["vanilla-fit", "--data", str(cases), "--preset", "cpu",
               "--out", str(bundle), "--epochs", "4", "--steps-per-epoch", "1",
               "--resume-from", str(bundle)])
    assert rc == 0
    record = json.loads((bundle / "checkpoint.json").read_text(encoding="utf-8"))
    assert record["epoch"] == 3
