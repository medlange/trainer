# SPDX-License-Identifier: Apache-2.0
"""DistributedDataParallel: the seam, its refusals, and one real two-process run.

THE SINGLE-PROCESS CLAIMS are environment-shape claims: no WORLD_SIZE (or a
world of one) means no process group is initialized — provable, because an
init attempt would need MASTER_ADDR and would raise; a WORLD_SIZE without a
RANK means a broken launch and gets a named refusal, not a rendezvous
timeout.

THE REAL RUN is a genuine two-process Gloo world on CPU (torch.multiprocessing
.spawn, one training step each) asserting both ranks complete and rank 0's
recorded loss is finite. It is marked skip-if-not-enabled because CI is a
single machine and spawning processes is a dev-machine activity — set
MEDOS_RUN_DDP=1 to run it:

    MEDOS_RUN_DDP=1 pytest tests/test_distributed.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from medos_trainer.vanilla.distributed import (
    get_rank,
    is_main_process,
    maybe_init_distributed,
)


def test_no_world_size_means_no_distribution(monkeypatch) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    assert maybe_init_distributed("cpu") is False
    assert get_rank() == 0
    assert is_main_process()


def test_a_world_of_one_is_not_distributed(monkeypatch) -> None:
    """WORLD_SIZE=1 + RANK=0 must still return False — and the proof that no
    process group was initialized is that initializing one would need
    MASTER_ADDR, which this environment deliberately does not have: any init
    attempt would raise."""
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    assert maybe_init_distributed("cpu") is False
    assert get_rank() == 0


def test_world_size_without_rank_is_refused_named(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.delenv("RANK", raising=False)
    with pytest.raises(RuntimeError, match="RANK is not set"):
        maybe_init_distributed("cpu")


def test_garbage_world_size_treats_the_world_as_one(monkeypatch) -> None:
    """A corrupt launcher that writes WORLD_SIZE=banana must not crash the
    single-process path — unparseable is the same as absent."""
    monkeypatch.setenv("WORLD_SIZE", "banana")
    assert maybe_init_distributed("cpu") is False


def _ddp_worker(rank: int, world_size: int, port: int, tmp: str) -> None:
    """One child of the real two-process run: build the world by hand (a bare
    mp.spawn has no torchrun), fit ONE step on the toy, and have rank 0 leave
    its recorded loss on disk for the parent to judge."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    from medos_trainer.vanilla.distributed import shutdown_distributed
    from medos_trainer.vanilla.nets import UNetConfig, build_unet
    from medos_trainer.vanilla.trainer import FitPlan, VanillaTrainer
    from test_vanilla_data import _toy_cases

    torch.manual_seed(0)
    net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                features=(4, 8, 16), deep_supervision=True))
    plan = FitPlan(patch_size=(16, 16, 16), batch_size=2, steps_per_epoch=1,
                   epochs=1, foreground_prob=1.0)
    trainer = VanillaTrainer(net, num_classes=2, plan=plan, device="cpu")
    assert trainer.distributed, "the child must have joined the world"

    train, val = _toy_cases(2, seed=0), _toy_cases(1, seed=100)
    # The contract the trainer documents: each rank's stream is seed+rank.
    result = trainer.fit(train, val, np.random.default_rng(0 + rank))
    if rank == 0:
        assert len(result["history"]) == 1
        (Path(tmp) / "rank0.json").write_text(
            json.dumps(result["history"][0]), encoding="utf-8"
        )
    # Both ranks rendezvous before teardown so neither outruns the other's
    # broadcast, then the group is destroyed before the child exits (a clean
    # gloo teardown on every platform, Windows included).
    shutdown_distributed()


@pytest.mark.skipif(
    os.environ.get("MEDOS_RUN_DDP") != "1",
    reason="real two-process DDP run; enable with MEDOS_RUN_DDP=1 on a dev machine",
)
def test_real_two_process_gloo_run_completes_with_finite_loss(tmp_path) -> None:
    import socket

    import torch.multiprocessing as mp

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    mp.spawn(_ddp_worker, args=(2, port, str(tmp_path)), nprocs=2, join=True)

    record = json.loads((tmp_path / "rank0.json").read_text(encoding="utf-8"))
    assert np.isfinite(record["loss"])
    assert np.isfinite(record["val_masked_dice_loss"])
