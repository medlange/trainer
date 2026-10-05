# SPDX-License-Identifier: Apache-2.0
"""Sliding-window inference and the predict path for the vanilla stack."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from medos_trainer.vanilla.data import Case
from medos_trainer.vanilla.infer import (
    SlidingWindowPredictor,
    gaussian_window,
    load_predictor,
    save_inference_bundle,
    served_net,
    window_starts,
)
from medos_trainer.vanilla.nets import UNetConfig, build_unet
from medos_trainer.vanilla.trainer import FitPlan, VanillaTrainer


def test_window_starts_covers_the_volume_exactly() -> None:
    # Shorter than the patch: a single window at 0.
    assert window_starts(10, 16, 0.5) == [0]
    # An exact stride fit: 0, 8, 16 with the flush window already present.
    assert window_starts(32, 16, 0.5) == [0, 8, 16]
    # A ragged tail still gets its last window at length - patch.
    starts = window_starts(40, 16, 0.5)
    assert starts[-1] == 24
    assert sorted(starts) == [0, 8, 16, 24]
    # Overlap bounds are enforced by the predictor, but starts stay sane at 0.
    assert window_starts(64, 16, 0.0)[1] - window_starts(64, 16, 0.0)[0] == 16


def test_gaussian_window_is_peaked_and_symmetric() -> None:
    w = gaussian_window((16, 16, 16))
    assert w.shape == (16, 16, 16)
    assert float(w.max()) == pytest.approx(1.0)
    assert float(w[7, 7, 7]) == pytest.approx(1.0, abs=0.1)
    assert np.allclose(w.numpy(), np.flip(w.numpy(), axis=(0, 1, 2)))
    # Edges are down-weighted relative to the centre.
    assert float(w[0, 7, 7]) < 0.5


def _tiny_net() -> torch.nn.Module:
    torch.manual_seed(0)
    return build_unet(UNetConfig(input_channels=1, num_classes=2,
                                 features=(4, 8, 16), deep_supervision=False))


def test_patch_equal_to_volume_matches_direct_forward() -> None:
    """One window covering the whole volume must equal the plain forward pass."""
    net = _tiny_net()
    predictor = SlidingWindowPredictor(net, patch_size=(16, 16, 16), overlap=0.5)
    image = np.random.default_rng(0).normal(size=(1, 16, 16, 16)).astype(np.float32)
    label, probs = predictor.predict(image)
    with torch.no_grad():
        direct = torch.softmax(net(torch.as_tensor(image)[None]), dim=1)[0]
    assert np.allclose(probs, direct.numpy(), atol=1e-5)
    assert np.array_equal(label, direct.argmax(dim=0).numpy())


def test_predictor_refuses_the_training_net() -> None:
    training_net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                         features=(4, 8, 16), deep_supervision=True))
    with pytest.raises(ValueError, match="deep_supervision=False"):
        SlidingWindowPredictor(training_net, patch_size=(16, 16, 16))


def test_served_net_drops_only_aux_heads() -> None:
    training_net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                         features=(4, 8, 16), deep_supervision=True))
    twin = served_net(training_net)
    assert not twin.config.deep_supervision
    out = twin(torch.randn(1, 1, 16, 16, 16))
    assert isinstance(out, torch.Tensor), "served net returns one tensor"
    assert out.shape == (1, 2, 16, 16, 16)
    # The training net still returns its deep-supervision tuple, last at full res.
    training_net.eval()
    with torch.no_grad():
        full = training_net(torch.randn(1, 1, 16, 16, 16))
    assert isinstance(full, tuple)
    assert full[-1].shape == (1, 2, 16, 16, 16)


def test_bundle_round_trips_through_load_predictor(tmp_path) -> None:
    torch.manual_seed(0)
    net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                features=(4, 8, 16), deep_supervision=True))
    record = {"epoch": 2, "loss": 0.4, "val_masked_dice_loss": 0.8, "lr": 0.01}
    save_inference_bundle(tmp_path, net, record, patch_size=(16, 16, 16))

    assert json.loads((tmp_path / "net_config.json").read_text())["deep_supervision"]
    plan = json.loads((tmp_path / "fit_plan.json").read_text())
    assert plan["patch_size"] == [16, 16, 16]
    saved_record = json.loads((tmp_path / "checkpoint.json").read_text())
    assert saved_record == record

    predictor = load_predictor(tmp_path)
    assert predictor.patch_size == (16, 16, 16)
    assert not predictor.net.config.deep_supervision
    image = np.random.default_rng(1).normal(size=(1, 20, 20, 20)).astype(np.float32)
    label, probs = predictor.predict(image)
    assert label.shape == (20, 20, 20)
    assert probs.shape == (2, 20, 20, 20)


def test_end_to_end_train_bundle_predict(tmp_path) -> None:
    """plan -> fit -> export -> load_predictor must segment the toy it trained on."""
    from test_vanilla_data import _toy_cases

    def dice(case: Case, label: np.ndarray) -> float:
        fg = (case.label == 1) & (case.mask[0] > 0)
        return 2.0 * float((label == 1)[fg].sum()) / float((label == 1).sum() + fg.sum())

    torch.manual_seed(0)
    net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                features=(4, 8, 16), deep_supervision=True))
    plan = FitPlan(patch_size=(16, 16, 16), batch_size=2, steps_per_epoch=8,
                   epochs=8, foreground_prob=1.0)
    trainer = VanillaTrainer(net, num_classes=2, plan=plan)
    train, val = _toy_cases(4, seed=0), _toy_cases(2, seed=100)
    trainer.fit(train, val, np.random.default_rng(0), out_dir=tmp_path)

    case = _toy_cases(1, seed=7)[0]
    label, _ = load_predictor(tmp_path).predict(case.image)

    # THE PIPELINE is what is under test, so the assertion is "learning
    # happened", not an absolute quality bar: clearly above both the untrained
    # net on the same case and the empty-prediction floor.
    untrained, _ = load_predictor(_untrained_bundle(tmp_path)).predict(case.image)
    learned, naive = dice(case, label), dice(case, untrained)
    assert learned > 0.3, f"predictor did not learn the toy: dice={learned}"
    assert learned > 2.0 * naive, f"no improvement over untrained: {learned} vs {naive}"


def _untrained_bundle(tmp_path) -> Path:
    """A second bundle, same config, never trained — the baseline's checkpoint dir."""
    out = tmp_path / "untrained"
    out.mkdir()
    fresh = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                  features=(4, 8, 16), deep_supervision=True))
    save_inference_bundle(out, fresh, {"epoch": -1, "loss": 0.0,
                                       "val_masked_dice_loss": 0.0, "lr": 0.01},
                          patch_size=(16, 16, 16))
    return out


def test_predict_cli_smoke(tmp_path) -> None:
    from test_vanilla_data import _toy_cases

    net = _tiny_net()
    bundle = tmp_path / "bundle"
    save_inference_bundle(bundle, net, {"epoch": 0, "loss": 1.0,
                                        "val_masked_dice_loss": 0.0, "lr": 0.01},
                          patch_size=(16, 16, 16))
    case_path = tmp_path / "case.npz"
    case = _toy_cases(1, seed=11)[0]
    np.savez(case_path, image=case.image, label=case.label, mask=case.mask)
    out_path = tmp_path / "pred.npz"

    from medos_trainer.__main__ import main

    rc = main(["predict", "--checkpoint-dir", str(bundle),
               "--input", str(case_path), "--output", str(out_path)])
    assert rc == 0
    with np.load(out_path) as z:
        assert z["label"].shape == case.label.shape
        assert z["probabilities"].shape == (2,) + case.label.shape
