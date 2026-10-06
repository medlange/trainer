# SPDX-License-Identifier: Apache-2.0
"""Cascades: the coarse probability map rides in as a fine-model channel.

THE PROPERTIES UNDER TEST: `build_cascade_cases` appends exactly one image
channel (the coarse foreground probability) while leaving label, spacing and
case_id untouched and keeping the mask shape-consistent; the fine fit runs
end to end with the enlarged channel count and its bundle loads through
`load_predictor` and predicts on cascade-shaped input; the command-line
`--cascade-from` writes `fine/` + `cascade.json` and the JSON parses with
the keys a consumer needs.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from medos_trainer.__main__ import main
from medos_trainer.standalone import fit_cascade_command, fit_command
from medos_trainer.vanilla.cascade import build_cascade_cases
from medos_trainer.vanilla.infer import load_predictor
from medos_trainer.vanilla.nets import UNetConfig, build_unet
from medos_trainer.vanilla.trainer import FitPlan, VanillaTrainer


def _coarse_bundle(tmp_path) -> object:
    """A genuinely trained tiny coarse model on the toy — the cascade's input
    must be a real probability map, not a scratch init's noise."""
    from test_vanilla_data import _toy_cases

    torch.manual_seed(0)
    net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                features=(4, 8, 16), deep_supervision=True))
    plan = FitPlan(patch_size=(16, 16, 16), batch_size=2, steps_per_epoch=4,
                   epochs=4, foreground_prob=1.0)
    trainer = VanillaTrainer(net, num_classes=2, plan=plan)
    train, val = _toy_cases(4, seed=0), _toy_cases(2, seed=100)
    trainer.fit(train, val, np.random.default_rng(0), out_dir=tmp_path / "coarse")
    return tmp_path / "coarse"


def test_build_cascade_cases_appends_one_foreground_channel(tmp_path) -> None:
    from test_vanilla_data import _toy_cases

    coarse = load_predictor(_coarse_bundle(tmp_path))
    cases = _toy_cases(3, seed=5)
    cascaded = build_cascade_cases(coarse, cases)
    assert len(cascaded) == len(cases)
    for original, built in zip(cases, cascaded):
        assert built.image.shape[0] == original.image.shape[0] + 1
        assert built.image.shape[1:] == original.image.shape[1:]
        # everything but the image (and its mask's extra channel) is untouched
        assert np.array_equal(built.label, original.label)
        assert built.spacing_mm == original.spacing_mm
        assert built.case_id == original.case_id
        assert built.mask is not None and original.mask is not None
        assert built.mask.shape == built.image.shape
        assert np.array_equal(built.mask[:-1], original.mask)
        assert np.all(built.mask[-1] == 1.0), (
            "the guide channel is computed, hence supervised, everywhere"
        )
        # the original channels ride along unchanged, the guide is a probability
        assert np.array_equal(built.image[:-1], original.image)
        guide = built.image[-1]
        assert np.all(guide >= 0.0) and np.all(guide <= 1.0)


def test_fine_cascade_fit_end_to_end_predicts_with_enlarged_channels(tmp_path) -> None:
    from test_vanilla_data import _toy_cases

    coarse_dir = _coarse_bundle(tmp_path)
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    for i, case in enumerate(_toy_cases(5, seed=0)):
        np.savez(cases_dir / f"case-{i}.npz", image=case.image, label=case.label,
                 mask=case.mask)

    out_dir = tmp_path / "cascade"
    summary = fit_cascade_command(cases_dir, "cpu", out_dir, coarse_dir,
                                  epochs=2, steps_per_epoch=2)
    fine_dir = out_dir / "fine"
    assert (fine_dir / "model.pt").is_file()
    assert (fine_dir / "net_config.json").is_file()

    document = json.loads((out_dir / "cascade.json").read_text(encoding="utf-8"))
    assert document["coarse"] == str(coarse_dir)
    assert document["input_channels"] == 2
    assert document["fit"]["history"], "the fit summary carries the run"
    assert summary["best_val_masked_dice_loss"] >= 0.0

    # THE FINE BUNDLE was rebuilt by load_predictor with the WIDER stem, and
    # it predicts on cascade-shaped input — that is the whole contract.
    predictor = load_predictor(fine_dir)
    assert predictor.net.config.input_channels == 2
    case = _toy_cases(1, seed=7)[0]
    coarse = load_predictor(coarse_dir)
    built = build_cascade_cases(coarse, [case])[0]
    label, probs = predictor.predict(built.image)
    assert label.shape == case.label.shape
    assert probs.shape == (2,) + case.label.shape


def test_fit_command_cascade_from_delegates(tmp_path) -> None:
    from test_vanilla_data import _toy_cases

    coarse_dir = _coarse_bundle(tmp_path)
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    for i, case in enumerate(_toy_cases(4, seed=0)):
        np.savez(cases_dir / f"case-{i}.npz", image=case.image, label=case.label,
                 mask=case.mask)
    out_dir = tmp_path / "out"
    fit_command(cases_dir, "cpu", out_dir, epochs=1, steps_per_epoch=1,
                cascade_from=coarse_dir)
    assert (out_dir / "fine" / "model.pt").is_file()
    document = json.loads((out_dir / "cascade.json").read_text(encoding="utf-8"))
    assert document["input_channels"] == 2
    assert document["coarse"] == str(coarse_dir)


def test_cascade_cli_smoke(tmp_path) -> None:
    from test_vanilla_data import _toy_cases

    coarse_dir = _coarse_bundle(tmp_path)
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    for i, case in enumerate(_toy_cases(4, seed=0)):
        np.savez(cases_dir / f"case-{i}.npz", image=case.image, label=case.label,
                 mask=case.mask)
    out_dir = tmp_path / "out"
    rc = main(["vanilla-fit", "--data", str(cases_dir), "--preset", "cpu",
               "--out", str(out_dir), "--epochs", "1", "--steps-per-epoch", "1",
               "--cascade-from", str(coarse_dir)])
    assert rc == 0
    document = json.loads((out_dir / "cascade.json").read_text(encoding="utf-8"))
    assert document["input_channels"] == 2
    assert (out_dir / "fine" / "model.pt").is_file()
