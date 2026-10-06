# SPDX-License-Identifier: Apache-2.0
"""The poly learning-rate law, against the plateau default it joins.

TWO CLAIMS: (1) `FitPlan` refuses a schedule it does not know, naming the
choices, at construction — a frozen record must never hold an invalid law;
(2) on a poly run the recorded lr is what nnU-Net's PolyLRScheduler shape
promises: the first step sits at exactly `learning_rate`, the sequence is
strictly decreasing, and by the end of the run `(1 - progress)^0.9` has
dragged it below five percent of the base. The plateau path's byte-identical
behaviour is pinned by the pre-existing suite (resume, evaluate, standalone)
and is deliberately not re-asserted here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from medos_trainer.vanilla.nets import UNetConfig, build_unet
from medos_trainer.vanilla.trainer import FitPlan, VanillaTrainer


def test_fit_plan_refuses_an_unknown_schedule_naming_the_choices() -> None:
    with pytest.raises(ValueError, match=r"\('plateau', 'poly'\)"):
        FitPlan(patch_size=(16, 16, 16), lr_schedule="cosine")
    # both legal values construct; the default stays the historical plateau
    assert FitPlan(patch_size=(16, 16, 16)).lr_schedule == "plateau"
    assert FitPlan(patch_size=(16, 16, 16), lr_schedule="poly").lr_schedule == "poly"


def test_poly_lr_is_strictly_decreasing_from_base_to_below_five_percent() -> None:
    """The recorded lr is what nnU-Net's PolyLRScheduler shape promises: the
    first record sits at exactly `learning_rate`, the sequence is strictly
    decreasing, and once progress nears 1 the `(1 - progress)^0.9` factor has
    dragged the last record below five percent of the base.

    THE RUN SHAPE MATTERS, and the comment says why rather than hiding it: a
    record is written per EPOCH but the law moves per STEP, so with S steps
    per epoch the epoch-0 record already reflects progress (S-1)/(E·S) — the
    "first record == learning_rate" claim only holds untruncated when the
    epoch-0 step that set it sat at progress 0, i.e. one step per epoch. At
    one step per epoch the record of epoch e is exactly the law at e/E, so 50
    epochs put the last record at (1/50)^0.9 ≈ 0.03 of the base — under the
    five-percent bar. (The spec's sketch of a 2-epoch run cannot satisfy all
    three claims at once: at E=2 the deepest reachable progress is < 1 and
    (1/2)^0.9 ≈ 0.54. The formula is the contract; the claims are verified at
    a shape where they are arithmetically true.)"""
    from test_vanilla_data import _toy_cases

    torch.manual_seed(0)
    net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                features=(4, 8, 16), deep_supervision=True))
    plan = FitPlan(patch_size=(16, 16, 16), batch_size=2, steps_per_epoch=1,
                   epochs=50, foreground_prob=1.0, lr_schedule="poly")
    trainer = VanillaTrainer(net, num_classes=2, plan=plan)

    train, val = _toy_cases(4, seed=0), _toy_cases(2, seed=100)
    result = trainer.fit(train, val, np.random.default_rng(0))

    lrs = [record["lr"] for record in result["history"]]
    assert len(lrs) == 50
    assert lrs[0] == pytest.approx(plan.learning_rate)
    assert all(later < earlier for earlier, later in zip(lrs, lrs[1:])), lrs
    assert lrs[-1] < plan.learning_rate * 0.05, lrs
    # And the poly plan owns no scheduler object — the law is the step math.
    assert trainer.scheduler is None


def test_poly_lr_matches_the_closed_form_at_every_record() -> None:
    """The recorded value is the law evaluated at the epoch's LAST step, not
    an approximation — pin the exact numbers so a silent reorder of the
    lr update (e.g. after validation) shows up as a failure, not a drift."""
    from test_vanilla_data import _toy_cases

    torch.manual_seed(0)
    net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                features=(4, 8, 16), deep_supervision=True))
    steps, epochs = 3, 2
    plan = FitPlan(patch_size=(16, 16, 16), batch_size=1, steps_per_epoch=steps,
                   epochs=epochs, foreground_prob=1.0, lr_schedule="poly",
                   learning_rate=0.02)
    trainer = VanillaTrainer(net, num_classes=2, plan=plan)
    train, val = _toy_cases(4, seed=0), _toy_cases(2, seed=100)
    result = trainer.fit(train, val, np.random.default_rng(0))

    for record in result["history"]:
        epoch = record["epoch"]
        last_step = steps - 1
        progress = (epoch + last_step / steps) / epochs
        assert record["lr"] == pytest.approx(
            plan.learning_rate * (1.0 - progress) ** 0.9, rel=1e-6)


def test_poly_resume_record_carries_no_scheduler_and_restores(tmp_path) -> None:
    """The poly resume record is honest about owning no scheduler: saving and
    loading round-trips the net and optimizer and leaves the (nonexistent)
    scheduler alone, without inventing plateau state a poly run never had."""
    from test_vanilla_data import _toy_cases

    torch.manual_seed(0)
    net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                features=(4, 8, 16), deep_supervision=True))
    plan = FitPlan(patch_size=(16, 16, 16), batch_size=2, steps_per_epoch=2,
                   epochs=2, foreground_prob=1.0, lr_schedule="poly")
    trainer = VanillaTrainer(net, num_classes=2, plan=plan)
    train, val = _toy_cases(4, seed=0), _toy_cases(2, seed=100)
    trainer.fit(train, val, np.random.default_rng(0))
    state_path = trainer.save_state(tmp_path, epoch=1)
    saved = torch.load(state_path, map_location="cpu")
    assert saved["scheduler"] is None

    fresh = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                  features=(4, 8, 16), deep_supervision=True))
    resumed = VanillaTrainer(fresh, num_classes=2, plan=plan)
    restored = resumed.load_state(tmp_path)
    assert restored["epoch"] == 1
    for (n_a, p_a), (n_b, p_b) in zip(
        trainer.net.named_parameters(), resumed.net.named_parameters()
    ):
        assert n_a == n_b
        assert torch.equal(p_a, p_b)
