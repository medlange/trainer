# SPDX-License-Identifier: Apache-2.0
"""Multi-channel (C > 1) volumes through the whole pipeline.

The format contract says `image` is (C, K, J, I); until now only C=1 had
been exercised end to end. The toy makes the second channel MEANINGFUL:
class 1 is the rule "channel 0 bright AND channel 1 mid-range" — a
single-channel net cannot express it, so the pipeline under test genuinely
carries two channels through the fingerprint, the plan, the fit and the
served prediction.
"""

from __future__ import annotations

import numpy as np
import torch
from medos_trainer.vanilla.data import Case
from medos_trainer.vanilla.infer import load_predictor
from medos_trainer.vanilla.nets import UNetConfig, build_unet
from medos_trainer.vanilla.plan import collect_fingerprint, plan_from_fingerprint
from medos_trainer.vanilla.trainer import FitPlan, VanillaTrainer


def _two_channel_cases(n: int = 4, seed: int = 0) -> list[Case]:
    """ch0 = 200 inside ball A; ch1 = 100 inside ball B; label = A AND B.
    The balls sit nearly concentric (offset at most one voxel) so the AND
    region is most of each ball — a budget a CPU test can afford — while the
    rule still genuinely needs BOTH channels: outside either ball that
    channel is background, so a single-channel net cannot separate the
    intersection from the crescents."""
    rng = np.random.default_rng(seed)
    cases = []
    kk, jj, ii = np.ogrid[:24, :24, :24]
    for i in range(n):
        c_a = rng.integers(9, 14, size=3)
        c_b = c_a + rng.integers(-1, 2, size=3)
        ball_a = (kk - c_a[0]) ** 2 + (jj - c_a[1]) ** 2 + (ii - c_a[2]) ** 2 <= 7 ** 2
        ball_b = (kk - c_b[0]) ** 2 + (jj - c_b[1]) ** 2 + (ii - c_b[2]) ** 2 <= 7 ** 2
        image = np.stack([
            np.where(ball_a, 200.0, -600.0),
            np.where(ball_b, 100.0, -600.0),
        ]).astype(np.float32)
        image += rng.normal(0.0, 20.0, image.shape).astype(np.float32)
        label = (ball_a & ball_b).astype(np.int64)
        mask = np.ones_like(image)
        mask[:, :12, :12, :12] = 0.0  # an unlabelled corner quadrant
        cases.append(Case(image=image, label=label, mask=mask,
                          spacing_mm=(1.0, 1.0, 1.0), case_id=f"mc-{i}"))
    return cases


def _foreground_dice(case: Case, label: np.ndarray) -> float:
    fg = (case.label == 1) & (case.mask[0] > 0)
    denom = float((label == 1).sum() + fg.sum())
    return 2.0 * float((label == 1)[fg].sum()) / denom if denom else 0.0


def test_plan_builds_a_two_channel_config() -> None:
    cases = _two_channel_cases(3)
    fingerprint = collect_fingerprint(cases)
    plan = plan_from_fingerprint(fingerprint, "cpu")
    config = plan.network_config(input_channels=cases[0].image.shape[0])
    assert config.input_channels == 2
    fit = plan.fit_plan()
    assert fit.patch_size == plan.patch_size


def test_two_channel_end_to_end_learns_the_and_rule(tmp_path) -> None:
    torch.manual_seed(0)
    train, val = _two_channel_cases(4, seed=0), _two_channel_cases(2, seed=100)
    plan = plan_from_fingerprint(collect_fingerprint(train), "cpu")
    net = build_unet(plan.network_config(input_channels=2))
    trainer = VanillaTrainer(net, num_classes=2,
                             plan=FitPlan(patch_size=plan.patch_size,
                                          batch_size=2, steps_per_epoch=8,
                                          epochs=12, foreground_prob=1.0),
                             device="cpu")
    before = trainer.validate(val, np.random.default_rng(11))
    result = trainer.fit(train, val, np.random.default_rng(0), out_dir=tmp_path)

    case = _two_channel_cases(1, seed=7)[0]
    predictor = load_predictor(tmp_path)
    label, probs = predictor.predict(case.image)
    assert label.shape == case.label.shape
    assert probs.shape == (2,) + case.label.shape

    # As in the single-channel end-to-end: the assertion is "learning
    # happened", not an absolute quality bar.
    untrained, _ = load_predictor(_untrained_bundle(tmp_path)).predict(case.image)
    learned, naive = _foreground_dice(case, label), _foreground_dice(case, untrained)
    assert result["best_val_masked_dice_loss"] < before
    assert learned > 0.3, f"two-channel pipeline did not learn the AND rule: {learned}"
    assert learned > 2.0 * naive, f"no improvement over untrained: {learned} vs {naive}"


def _untrained_bundle(tmp_path):
    """A never-trained twin bundle — the baseline's checkpoint dir."""
    from pathlib import Path

    from medos_trainer.vanilla.infer import save_inference_bundle

    out = Path(tmp_path) / "untrained"
    out.mkdir()
    fresh = build_unet(UNetConfig(input_channels=2, num_classes=2,
                                  features=(16, 32, 64, 128), deep_supervision=True))
    save_inference_bundle(out, fresh, {"epoch": -1, "loss": 0.0,
                                       "val_masked_dice_loss": 0.0, "lr": 0.01},
                          patch_size=(16, 16, 16))
    return out
