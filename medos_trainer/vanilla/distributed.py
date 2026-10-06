# SPDX-License-Identifier: Apache-2.0
"""Distributed helpers: the DDP seam, kept to four small functions.

THE WHOLE MECHANISM IS ENVIRONMENT-DRIVEN, because that is how `torchrun`
launches a world: `WORLD_SIZE`, `RANK`, `MASTER_ADDR` and `MASTER_PORT` are
set before the interpreter starts, and `maybe_init_distributed()` reads them.
A normal single-process run has none of that — `WORLD_SIZE` absent or 1 — and
the function returns False WITHOUT initializing anything, which keeps the
single-process path byte-identical to what it always was: no process group,
no wrapping, no synchronisation. A world of ONE is not distributed.

WHEN A WORLD IS REAL (WORLD_SIZE > 1), the trainer wraps its net in
`torch.nn.parallel.DistributedDataParallel`, only rank 0 validates, writes
checkpoints and prints, and the per-rank patch streams are seeded `seed+rank`
so no two ranks train on the same draws. THE SAMPLING IS WITH REPLACEMENT
(the sampler draws a fresh random patch per step), so overlapping draws
between ranks are expected and harmless — the ranks see different random
subsets of the same corpus, averaged by DDP's gradient all-reduce exactly as
a larger single-process batch would be.

THE BACKEND follows the device: NCCL for CUDA, Gloo for CPU — Gloo so that a
CPU-only dev box can run the real two-process path the tests exercise.
"""

from __future__ import annotations

import os

import torch.distributed as dist


def maybe_init_distributed(device: str = "cpu") -> bool:
    """Initialize the process group iff this process belongs to a real world.

    Returns True when the caller must run distributed (wrap the net, guard
    rank-0 work). Returns False for every single-process shape — WORLD_SIZE
    absent, "1", or unparseable — and in that case NO process group is
    initialized: the proof is that `init_process_group` would need
    MASTER_ADDR and would raise. WORLD_SIZE > 1 without RANK is a broken
    launch (something set the world but not the rank), refused with a named
    error rather than a rendezvous timeout. Already-initialized is idempotent
    True, so constructing a second trainer in one process is legal.
    """
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        world_size = 1
    if world_size <= 1:
        return False
    if dist.is_available() and dist.is_initialized():
        return True
    if os.environ.get("RANK") is None:
        raise RuntimeError(
            "distributed launch detected (WORLD_SIZE="
            f"{world_size}) but RANK is not set; launch through torchrun, "
            "e.g. torchrun --nproc_per_node=2 -m medos_trainer vanilla-fit ..."
        )
    backend = "nccl" if device.startswith("cuda") else "gloo"
    # env:// reads MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE — torchrun sets all
    # four; a bare mp.spawn child sets them itself (see tests/test_distributed.py).
    dist.init_process_group(backend)
    return True


def get_rank() -> int:
    """This process's rank, or 0 whenever no group is initialized."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    """Rank 0 does the validating, checkpointing and printing; every other
    rank trains. Without a group, every process is the main process."""
    return get_rank() == 0


def shutdown_distributed() -> None:
    """Tear the group down when the run is over. Idempotent, and a no-op on
    the single-process path — call sites never need to ask whether DDP ran."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
