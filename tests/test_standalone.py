# SPDX-License-Identifier: Apache-2.0
"""The autonomous entry: cases in, plan/fit/bundle out, no platform."""

from __future__ import annotations

import json

import numpy as np
import pytest
from medos_trainer.__main__ import main
from medos_trainer.standalone import (
    fit_command,
    import_nnunet_dataset,
    load_cases_dir,
    plan_command,
)


def _write_cases(tmp_path, n: int = 5, shape=(20, 20, 20), spacing=(1.0, 1.0, 1.0)):
    rng = np.random.default_rng(0)
    out = tmp_path / "cases"
    out.mkdir()
    for i in range(n):
        image = rng.normal(-600.0, 50.0, (1, *shape)).astype(np.float32)
        label = (rng.random(shape) < 0.05).astype(np.int64)
        np.savez(out / f"case-{i}.npz", image=image, label=label,
                 spacing_mm=np.asarray(spacing, dtype=np.float64))
    return out


def test_load_cases_dir_reads_spacing_and_sorts(tmp_path) -> None:
    out = _write_cases(tmp_path, spacing=(2.0, 0.7, 0.7))
    cases = load_cases_dir(out)
    assert [c.case_id for c in cases] == [f"case-{i}" for i in range(5)]
    assert cases[0].spacing_mm == (2.0, 0.7, 0.7)


def test_load_cases_dir_refuses_an_empty_directory(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no .npz"):
        load_cases_dir(empty)


def test_plan_command_writes_the_reasoned_plan(tmp_path) -> None:
    cases = _write_cases(tmp_path)
    out = tmp_path / "plan.json"
    document = plan_command(cases, "cpu", out)
    assert out.is_file()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == document
    assert document["preset"] == "cpu"
    assert document["reasons"], "the plan must say why"
    # spacing rides the npz member into the fingerprint, json round-trip and all
    assert document["median_spacing"] == [1.0, 1.0, 1.0]


def test_fit_command_trains_and_writes_a_bundle(tmp_path) -> None:
    cases = _write_cases(tmp_path)
    bundle = tmp_path / "bundle"
    summary = fit_command(cases, "cpu", bundle, epochs=2, steps_per_epoch=2)
    assert (bundle / "model.pt").is_file()
    assert (bundle / "net_config.json").is_file()
    assert (bundle / "fit_plan.json").is_file()
    assert summary["best_val_masked_dice_loss"] >= 0.0


def test_vanilla_fit_cli_smoke(tmp_path) -> None:
    cases = _write_cases(tmp_path)
    bundle = tmp_path / "bundle"
    rc = main(["vanilla-fit", "--data", str(cases), "--preset", "cpu",
               "--out", str(bundle), "--epochs", "1", "--steps-per-epoch", "1"])
    assert rc == 0
    assert (bundle / "model.pt").is_file()


def test_vanilla_plan_cli_smoke(tmp_path) -> None:
    cases = _write_cases(tmp_path)
    out = tmp_path / "plan.json"
    rc = main(["vanilla-plan", "--data", str(cases), "--preset", "cpu",
               "--out", str(out)])
    assert rc == 0
    assert json.loads(out.read_text(encoding="utf-8"))["preset"] == "cpu"


def test_import_nnunet_dataset_round_trip(tmp_path) -> None:
    nib = pytest.importorskip("nibabel")
    images = tmp_path / "imagesTr"
    labels = tmp_path / "labelsTr"
    images.mkdir()
    labels.mkdir()
    data = np.random.default_rng(0).normal(size=(10, 12, 14)).astype(np.float32)
    seg = (np.random.default_rng(1).random((10, 12, 14)) < 0.1).astype(np.int32)
    image = nib.Nifti1Image(data, np.eye(4))
    image.header.set_zooms((2.0, 0.7, 0.7))
    label_image = nib.Nifti1Image(seg, np.eye(4))
    label_image.header.set_zooms((2.0, 0.7, 0.7))
    nib.save(image, str(images / "case_000.nii.gz"))
    nib.save(label_image, str(labels / "case_000.nii.gz"))

    out = tmp_path / "cases"
    n = import_nnunet_dataset(images, labels, out)
    assert n == 1
    cases = load_cases_dir(out)
    assert cases[0].image.shape == (1, 10, 12, 14)
    # NIfTI stores zooms as float32; the spacing survives to format precision.
    assert cases[0].spacing_mm == pytest.approx((2.0, 0.7, 0.7), abs=1e-6)
    assert np.array_equal(cases[0].label, seg)


def test_import_nnunet_dataset_names_the_missing_label(tmp_path) -> None:
    nib = pytest.importorskip("nibabel")
    images = tmp_path / "imagesTr"
    labels = tmp_path / "labelsTr"
    images.mkdir()
    labels.mkdir()
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)),
             str(images / "lonely.nii.gz"))
    with pytest.raises(ValueError, match="lonely"):
        import_nnunet_dataset(images, labels, tmp_path / "out")
