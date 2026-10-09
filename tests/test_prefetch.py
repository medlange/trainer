# SPDX-License-Identifier: Apache-2.0
"""The batch prefetcher and its fit integration.

THREE CLAIMS: (1) `BatchPrefetcher` is a transparent pipe — the consumer sees
exactly the factory's items, in order, then StopIteration, no matter how small
the queue; (2) a factory that dies delivers its exception to the consumer
rather than hanging it; (3) a prefetching fit is a REAL fit — finite losses
that decrease, a checkpoint on disk, and a model that beats the untrained
baseline on the toy dice — and the inline path (prefetch_batches=0) keeps its
byte-identical behaviour, which the pre-existing suite pins.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from medos_trainer.vanilla.nets import UNetConfig, build_unet
from medos_trainer.vanilla.prefetch import BatchPrefetcher
from medos_trainer.vanilla.trainer import FitPlan, VanillaTrainer

TOY_NET = {"features": (4, 8, 16), "stem_stride": (1, 1, 1)}


def _batch(tag: int) -> tuple[np.ndarray, np.ndarray, None]:
    images = np.full((1, 1, 4, 4, 4), float(tag), dtype=np.float32)
    labels = np.full((1, 4, 4, 4), tag, dtype=np.int64)
    return images, labels, None


def test_prefetcher_yields_factory_items_in_order_then_stops() -> None:
    items = [_batch(0), _batch(1), _batch(2)]
    prefetcher = BatchPrefetcher(lambda: iter(items), queue_size=2)
    try:
        # list() consumes to StopIteration: the pipe is transparent, and the
        # sentinel (not a hang, not an extra item) ends it.
        seen = list(prefetcher)
        assert len(seen) == len(items)
        for got, want in zip(seen, items):
            assert np.array_equal(got[0], want[0])
            assert np.array_equal(got[1], want[1])
            assert got[2] is None and want[2] is None
        with pytest.raises(StopIteration):
            next(prefetcher)
    finally:
        prefetcher.close()


def test_prefetcher_surfaces_producer_exception() -> None:
    def factory():
        yield _batch(0)
        raise RuntimeError("producer exploded")

    prefetcher = BatchPrefetcher(factory, queue_size=2)
    try:
        assert next(prefetcher)[0][0, 0, 0, 0, 0] == 0.0
        # The exception arrives on the CONSUMER thread, type and message
        # intact — never swallowed into a silent StopIteration.
        with pytest.raises(RuntimeError, match="producer exploded"):
            next(prefetcher)
    finally:
        prefetcher.close()


def test_fit_plan_refuses_negative_prefetch_batches() -> None:
    with pytest.raises(ValueError, match="prefetch_batches"):
        FitPlan(patch_size=(16, 16, 16), prefetch_batches=-1)
    # 0 is the inline default every existing test relies on.
    assert FitPlan(patch_size=(16, 16, 16)).prefetch_batches == 0


def _fit_one(prefetch_batches: int, seed: int, out_dir) -> tuple[VanillaTrainer, dict]:
    """One toy run at the given queue depth; returns trainer and history."""
    from test_vanilla_data import _toy_cases

    torch.manual_seed(seed)
    net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                deep_supervision=True, **TOY_NET))
    plan = FitPlan(patch_size=(16, 16, 16), batch_size=2, steps_per_epoch=8,
                   epochs=4, foreground_prob=1.0,
                   prefetch_batches=prefetch_batches)
    trainer = VanillaTrainer(net, num_classes=2, plan=plan)
    train, val = _toy_cases(4, seed=0), _toy_cases(2, seed=100)
    result = trainer.fit(train, val, np.random.default_rng(seed), out_dir=out_dir)
    return trainer, result


def _val_loss(trainer: VanillaTrainer) -> float:
    from test_vanilla_data import _toy_cases

    return trainer.validate(_toy_cases(2, seed=100), np.random.default_rng(123))


def test_fit_with_prefetch_trains_and_writes_checkpoint(tmp_path) -> None:
    trainer, result = _fit_one(prefetch_batches=2, seed=0, out_dir=tmp_path / "bundle")
    losses = [record["loss"] for record in result["history"]]
    assert len(losses) == 4
    assert all(np.isfinite(losses)), losses
    assert losses[-1] < losses[0], losses
    assert (tmp_path / "bundle" / "model.pt").is_file()
    assert (tmp_path / "bundle" / "checkpoint.json").is_file()

    # The trained net beats an UNTRAINED twin on the toy dice: same init
    # (same torch seed), same validation patches (same validation rng).
    torch.manual_seed(0)
    untrained = VanillaTrainer(
        build_unet(UNetConfig(input_channels=1, num_classes=2,
                              deep_supervision=True, **TOY_NET)),
        num_classes=2,
        plan=FitPlan(patch_size=(16, 16, 16), batch_size=2, steps_per_epoch=1,
                     epochs=1, foreground_prob=1.0),
    )
    assert _val_loss(trainer) < _val_loss(untrained)


def test_prefetch_and_inline_runs_both_learn(tmp_path) -> None:
    """The prefetch path is an honest training path, not a different one:
    each arm's loss decreases against its own first epoch, and each arm's
    model beats the untrained baseline. Same seed across arms is NOT
    required — the two paths draw different sample streams BY DESIGN
    (documented on FitPlan.prefetch_batches), so the assertion is per-arm
    progress, not stream equality."""
    inline, inline_result = _fit_one(prefetch_batches=0, seed=1,
                                     out_dir=tmp_path / "inline")
    prefetch, prefetch_result = _fit_one(prefetch_batches=2, seed=1,
                                         out_dir=tmp_path / "prefetch")

    for name, result in (("inline", inline_result), ("prefetch", prefetch_result)):
        losses = [record["loss"] for record in result["history"]]
        assert all(np.isfinite(losses)), (name, losses)
        assert losses[-1] < losses[0], (name, losses)

    torch.manual_seed(1)
    untrained = VanillaTrainer(
        build_unet(UNetConfig(input_channels=1, num_classes=2,
                              deep_supervision=True, **TOY_NET)),
        num_classes=2,
        plan=FitPlan(patch_size=(16, 16, 16), batch_size=2, steps_per_epoch=1,
                     epochs=1, foreground_prob=1.0),
    )
    baseline = _val_loss(untrained)
    assert _val_loss(inline) < baseline
    assert _val_loss(prefetch) < baseline
