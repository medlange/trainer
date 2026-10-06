# SPDX-License-Identifier: Apache-2.0
"""Data and training for the vanilla stack, held to the contract."""

from __future__ import annotations

import json

import numpy as np
import torch
from medos_trainer.vanilla.data import (
    Case,
    PatchSampler,
    augment_mirror_rotate,
    load_case_npz,
    make_batch,
)
from medos_trainer.vanilla.losses import MaskedSegmentationLoss
from medos_trainer.vanilla.nets import UNetConfig, build_unet
from medos_trainer.vanilla.trainer import FitPlan, VanillaTrainer

TOY_PLAN = FitPlan(patch_size=(16, 16, 16), batch_size=2, steps_per_epoch=4,
                   epochs=3, foreground_prob=0.5)


def _toy_cases(n: int = 4, seed: int = 0) -> list[Case]:
    """A learnable toy: a bright sphere at a random centre is class 1; part of
    the voxels around it is unlabelled (mask 0) to exercise the masked path."""
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n):
        image = rng.normal(-600.0, 50.0, (1, 24, 24, 24)).astype(np.float32)
        label = np.zeros((24, 24, 24), dtype=np.int64)
        c = rng.integers(8, 16, size=3)
        kk, jj, ii = np.ogrid[:24, :24, :24]
        ball = (kk - c[0]) ** 2 + (jj - c[1]) ** 2 + (ii - c[2]) ** 2 <= 5 ** 2
        image[0][ball] = 200.0
        label[ball] = 1
        mask = np.ones_like(image)
        # Unlabel a corner quadrant: voxels there must not influence the loss.
        mask[:, :12, :12, :12] = 0.0
        cases.append(Case(image=image, label=label, mask=mask,
                          spacing_mm=(1.0, 1.0, 1.0), case_id=f"toy-{i}"))
    return cases


def test_masked_loss_ignores_unlabelled_voxels() -> None:
    case = _toy_cases(1, seed=3)[0]
    logits = torch.randn(1, 2, 24, 24, 24)
    base = MaskedSegmentationLoss(num_classes=2)(logits, torch.as_tensor(case.label)[None],
                                                 torch.as_tensor(case.mask))
    # Scramble every voxel the mask excludes: a masked loss must not move.
    label2 = case.label.copy()
    label2[case.mask[0] == 0] = (label2[case.mask[0] == 0] + 1) % 2
    after = MaskedSegmentationLoss(num_classes=2)(logits, torch.as_tensor(label2)[None],
                                                  torch.as_tensor(case.mask))
    assert torch.isclose(base, after)


def test_masked_loss_penalizes_labelled_errors() -> None:
    case = _toy_cases(1, seed=3)[0]
    target = torch.as_tensor(case.label)[None]
    mask = torch.as_tensor(case.mask)
    good = torch.zeros(1, 2, 24, 24, 24)
    good[:, 1] = 10.0 * (target == 1).float() - 10.0 * (target == 0).float()
    loss = MaskedSegmentationLoss(num_classes=2)(good, target, mask)
    assert float(loss) < 0.2


def test_sampler_shapes_and_foreground_bias() -> None:
    rng = np.random.default_rng(2)
    case = _toy_cases(1, seed=5)[0]
    sampler = PatchSampler((16, 16, 16), foreground_prob=1.0)
    patch = sampler.sample(case, rng)
    assert patch.image.shape == (1, 16, 16, 16)
    assert patch.label.shape == (16, 16, 16)
    assert patch.mask.shape == (1, 16, 16, 16)
    # With foreground_prob = 1 the centre sits on the labelled foreground.
    k, j, i = patch.centre
    assert case.label[k, j, i] == 1 and case.mask[0, k, j, i] == 1


def test_augment_preserves_shapes_and_values() -> None:
    rng = np.random.default_rng(4)
    case = _toy_cases(1, seed=7)[0]
    patch = PatchSampler((16, 16, 16)).sample(case, rng)
    aug = augment_mirror_rotate(patch, rng)
    assert aug.image.shape == patch.image.shape
    assert aug.label.shape == patch.label.shape
    # Mirroring/rotation permutes voxels; the multiset of image values is identical.
    assert np.isclose(np.sort(patch.image.ravel()), np.sort(aug.image.ravel())).all()


def test_augment_rotation_skipped_when_in_plane_is_not_square() -> None:
    """REAL-DATA REGRESSION, found by the PulmoAI benchmark: the plan sizes
    patches physically, so (K, J, I) = (16, 32, 16) is a legal planned patch;
    an unconditional 90-degree rotation then swapped the J/I extents, mixed
    (1,16,32,16) with (1,16,16,32) inside one batch, and `make_batch` raised.
    Mirrors never change extents and must still apply; the rotation is owed
    only to square in-plane patches; the batch the old code refused stacks."""
    case = _toy_cases(1, seed=11)[0]
    sampler = PatchSampler((8, 16, 8), foreground_prob=1.0)
    sources = [sampler.sample(case, np.random.default_rng(s)) for s in range(8)]
    augmented = [
        augment_mirror_rotate(p, np.random.default_rng(100 + s))
        for s, p in enumerate(sources)
    ]
    for source, aug in zip(sources, augmented):
        assert aug.image.shape == (1, 8, 16, 8)
        assert aug.label.shape == (8, 16, 8)
        assert aug.mask.shape == (1, 8, 16, 8)
        # mirrors/rotations permute voxels; the value multiset is identical.
        assert np.isclose(
            np.sort(source.image.ravel()), np.sort(aug.image.ravel())
        ).all()
    images, labels, _ = make_batch(augmented)
    assert images.shape == (8, 1, 8, 16, 8)
    assert labels.shape == (8, 8, 16, 8)


def test_case_round_trips_through_npz(tmp_path) -> None:
    case = _toy_cases(1, seed=9)[0]
    path = tmp_path / "case.npz"
    np.savez(path, image=case.image, label=case.label, mask=case.mask)
    loaded = load_case_npz(path, spacing_mm=(1.0, 1.0, 1.0))
    assert loaded.case_id == "case"
    assert np.array_equal(loaded.image, case.image)
    assert np.array_equal(loaded.mask, case.mask)


def test_trainer_learns_the_toy_and_checkpoints(tmp_path) -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                features=(4, 8, 16), deep_supervision=True))
    trainer = VanillaTrainer(net, num_classes=2, plan=TOY_PLAN)
    train, val = _toy_cases(4, seed=0), _toy_cases(2, seed=100)

    before = trainer.validate(val, np.random.default_rng(123))
    result = trainer.fit(train, val, rng, out_dir=tmp_path)
    after = result["best_val_masked_dice_loss"]
    # The masked soft-DICE LOSS goes down as the net learns the bright sphere.
    assert after < before, f"no progress: before={before}, after={after}"
    assert (tmp_path / "model.pt").is_file()
    record = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert {"epoch", "loss", "val_masked_dice_loss", "lr"} <= set(record)


def test_deep_supervision_tuple_flows_through_training() -> None:
    torch.manual_seed(0)
    net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                features=(4, 8, 16), deep_supervision=True))
    trainer = VanillaTrainer(net, num_classes=2, plan=TOY_PLAN)
    case = _toy_cases(1, seed=1)[0]
    images, labels, masks = make_batch([PatchSampler(TOY_PLAN.patch_size).sample(
        case, np.random.default_rng(0))])
    loss = trainer.train_step(torch.as_tensor(images), torch.as_tensor(labels),
                              torch.as_tensor(masks))
    assert np.isfinite(loss) and loss > 0

