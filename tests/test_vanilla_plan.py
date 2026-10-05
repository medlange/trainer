# SPDX-License-Identifier: Apache-2.0
"""Planning: the fingerprint decides the numbers, and says why."""

from __future__ import annotations

import numpy as np
import pytest
from medos_trainer.vanilla.data import Case
from medos_trainer.vanilla.plan import (
    PRESETS,
    VramPreset,
    collect_fingerprint,
    plan_from_fingerprint,
)


def _cases(spacing=(1.0, 1.0, 1.0), shape=(24, 32, 40), n: int = 3,
           seed: int = 0, with_mask: bool = True) -> list[Case]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        image = rng.normal(-600.0, 50.0, (1, *shape)).astype(np.float32)
        label = (rng.random(shape) < 0.05).astype(np.int64)
        mask = np.ones_like(image) if with_mask else None
        out.append(Case(image=image, label=label, mask=mask,
                        spacing_mm=spacing, case_id=f"c{i}"))
    return out


def test_fingerprint_medians_and_census() -> None:
    cases = _cases()
    fp = collect_fingerprint(cases)
    assert fp.median_shape == (24, 32, 40)
    assert fp.median_spacing == (1.0, 1.0, 1.0)
    assert len(fp.shapes) == 3 and len(fp.spacings) == 3
    # Background dominates; the census carries every present class.
    classes = dict(fp.class_counts)
    assert set(classes) == {0, 1}
    assert classes[0] > classes[1] > 0
    assert 0.0 < fp.foreground_fraction < 0.1
    assert fp.num_classes == 2
    # Fully labelled: every voxel counts as labelled evidence.
    assert fp.labelled_fraction == pytest.approx(1.0)


def test_fingerprint_counts_only_supervised_voxels() -> None:
    cases = _cases()
    case = cases[0]
    half = np.zeros_like(case.mask)
    half[:, :, :, :20] = 1.0  # label only half the volume
    cases[0] = Case(image=case.image, label=case.label, mask=half,
                    spacing_mm=case.spacing_mm, case_id=case.case_id)
    labelled_before = sum(int(c.mask.sum()) for c in cases if c.mask is not None)
    fp = collect_fingerprint(cases)
    assert fp.labelled_fraction == pytest.approx(labelled_before / (3 * 24 * 32 * 40))


def test_fingerprint_refuses_empty_corpus_and_unlabelled() -> None:
    with pytest.raises(ValueError, match="at least one case"):
        collect_fingerprint([])
    case = _cases(n=1)[0]
    unlabelled = Case(image=case.image, label=case.label, mask=np.zeros_like(case.mask),
                      spacing_mm=case.spacing_mm)
    with pytest.raises(ValueError, match="no labelled voxels"):
        collect_fingerprint([unlabelled])


def test_plan_isotropic_spacing_full_stem() -> None:
    fp = collect_fingerprint(_cases(spacing=(1.0, 1.0, 1.0)))
    plan = plan_from_fingerprint(fp, "cpu")
    # 12/1=12 -> 16, 20/1=20 -> 16: rounded to the stride multiple.
    assert all(p % 16 == 0 for p in plan.patch_size)
    assert plan.patch_size[0] == 16
    assert plan.stem_stride == (1, 1, 1)
    assert plan.preset.name == "cpu"
    assert plan.reasons, "the plan must record why"


def test_plan_anisotropic_spacing_stems_the_through_plane_axis() -> None:
    fp = collect_fingerprint(_cases(spacing=(3.0, 1.0, 1.0)))
    plan = plan_from_fingerprint(fp, "cpu")
    assert plan.stem_stride == (2, 1, 1)
    # 12/3 = 4 -> 16 after the floor; in-plane 20 -> 16.
    assert plan.patch_size == (16, 16, 16)
    assert any("stem_stride (2, 1, 1)" in r for r in plan.reasons)


def test_plan_caps_the_voxel_budget_and_says_so() -> None:
    fp = collect_fingerprint(_cases(spacing=(0.5, 0.5, 0.5)))
    tiny = VramPreset("tiny", features=(8, 16, 32), batch_size=1, voxel_budget=16 * 32 * 32)
    plan = plan_from_fingerprint(fp, tiny)
    count = plan.patch_size[0] * plan.patch_size[1] * plan.patch_size[2]
    assert count <= tiny.voxel_budget * 1.5, f"patch {plan.patch_size} over budget"
    assert any("budget" in r for r in plan.reasons)
    assert plan.preset.name == "tiny"


def test_plan_unknown_preset_is_a_lookup_error_with_choices() -> None:
    fp = collect_fingerprint(_cases())
    with pytest.raises(ValueError, match="unknown preset"):
        plan_from_fingerprint(fp, "quantum")


def test_planned_run_builds_the_training_pair() -> None:
    """The plan's whole point: a fit-ready (config, fit_plan) pair."""
    fp = collect_fingerprint(_cases())
    plan = plan_from_fingerprint(fp, "small")
    config = plan.network_config(input_channels=1)
    assert config.num_classes == fp.num_classes
    assert config.features == PRESETS["small"].features
    fit = plan.fit_plan()
    assert fit.patch_size == plan.patch_size
    assert fit.batch_size == PRESETS["small"].batch_size


def test_fingerprint_is_frozen() -> None:
    fp = collect_fingerprint(_cases())
    with pytest.raises(AttributeError):
        fp.median_shape = (1, 1, 1)  # type: ignore[misc]
