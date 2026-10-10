# SPDX-License-Identifier: Apache-2.0
"""Checkpoint selection by the deployment metric, not by a proxy.

THE LESSON THIS MODULE IS (docs/benchmark-pulmo-2026-10-07.md, "Long-run
continuation and the checkpoint-selection lesson", 2026-10-10): the 250-epoch
PulmoAI run kept improving the masked-patch validation proxy (0.3261 -> 0.3138
at epoch 121) while the full-volume foreground Dice of the very bundle that
proxy selected FELL to 0.438 — and that worse bundle had OVERWRITTEN the
epoch-111 bundle's 0.539. A proxy that can anti-correlate with the deployment
metric on the very next epoch must not be what selects the checkpoint.

`volume_dice` is the selector's answer: FULL-VOLUME foreground Dice on the
validation split, the same definition the evaluator reports (`standalone.
_dice_rows`): sliding-window inference over the whole case, per-class Dice
restricted to SUPERVISED voxels, foreground mean over classes 1..C-1, then the
mean over cases. HIGHER is better — the opposite orientation of the patch
loss, so `trainer.fit` compares selection scores with `>` in both modes.

The score is computed on the cases `fit` receives, with NO preprocessing
replay: `fit_command` feeds the selector the already-resampled, already
z-scored training split, so selection ranks epochs on the training grid.
The deployment evaluator scores raw cases on their own grids through the
bundle's `preprocess.json`; that grid difference is second-order and identical
for every epoch, which is all a selector needs — a faithful RANKING signal,
not a re-run of the report.

PERF HONESTY: full-volume sliding-window inference over `selection_cases`
volumes is minutes per epoch at real CT sizes, next to a patch validation that
is seconds. Planned runs accept that cost deliberately (the benchmark paid
0.539 -> 0.438 for the cheap proxy); tests use tiny volumes, where it is
seconds.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from medos_trainer.vanilla.data import Case
from medos_trainer.vanilla.infer import gaussian_window, window_starts
from medos_trainer.vanilla.nets import VanillaUNet

#: Windows per forward inside the selector's sliding window — the same small
#: batch `SlidingWindowPredictor` uses; selection throughput is dominated by
#: the window count, not the batching.
_WINDOW_BATCH = 2


@torch.no_grad()
def volume_dice(
    net: VanillaUNet,
    cases: list[Case],
    device: str,
    max_cases: int,
    patch_size: tuple[int, int, int],
    overlap: float = 0.5,
) -> float:
    """Full-volume foreground Dice of `net` over up to `max_cases` cases.

    THE SERVED-STYLE FORWARD: `net` may be the TRAINING net (deep supervision
    on — the tuple's LAST element is the full-resolution head, exactly what
    the served twin keeps), so the selector scores the weights the optimizer
    just stepped, with no twin construction per epoch. Windows, Gaussian
    blending and coverage are `infer`'s own primitives — the same arithmetic
    the deployed predictor runs, which is the point: the selector optimizes
    what the deployment metric measures.

    Returns the mean over cases of the per-case foreground mean (classes
    1..C-1, supervised voxels only) — `standalone._dice_rows`' aggregate
    definition, so a selection score of 0.55 IS "the evaluator would report
    0.55 on these cases". HIGHER is better; 0.0 when no case is given.
    """
    selected = list(cases)[: int(max_cases)]
    if not selected:
        return 0.0
    num_classes = int(net.config.num_classes)
    window_weight = gaussian_window(patch_size).to(device)
    was_training = net.training
    net.eval()
    try:
        per_case: list[float] = []
        for case in selected:
            x = torch.as_tensor(
                case.image, dtype=torch.float32, device=device
            ).unsqueeze(0)
            shape = tuple(x.shape[-3:])
            starts = [
                window_starts(n, p, overlap) for n, p in zip(shape, patch_size)
            ]
            probs_sum = torch.zeros(num_classes, *shape, device=device)
            weight_sum = torch.zeros(*shape, device=device)
            coords = [
                (k, j, i) for k in starts[0] for j in starts[1] for i in starts[2]
            ]
            for b in range(0, len(coords), _WINDOW_BATCH):
                batch_coords = coords[b : b + _WINDOW_BATCH]
                patches = torch.stack(
                    [
                        x[
                            0,
                            :,
                            k : k + patch_size[0],
                            j : j + patch_size[1],
                            i : i + patch_size[2],
                        ]
                        for k, j, i in batch_coords
                    ]
                )
                out = net(patches)
                if isinstance(out, tuple):
                    # THE TRAINING NET'S DEEP-SUPERVISION TUPLE: the last
                    # element is the full-resolution head — the served twin's
                    # single output, taken here without building the twin.
                    out = out[-1]
                probs = F.softmax(out, dim=1)
                for n, (k, j, i) in enumerate(batch_coords):
                    sl = (
                        slice(k, k + patch_size[0]),
                        slice(j, j + patch_size[1]),
                        slice(i, i + patch_size[2]),
                    )
                    probs_sum[(slice(None),) + sl] += probs[n] * window_weight
                    weight_sum[sl] += window_weight
            predicted = (
                (probs_sum / weight_sum.unsqueeze(0)).argmax(dim=0).cpu().numpy()
            )
            per_case.append(_case_foreground_dice(predicted, case, num_classes))
    finally:
        if was_training:
            net.train()
    return float(np.mean(per_case)) if per_case else 0.0


def _case_foreground_dice(
    predicted: np.ndarray, case: Case, num_classes: int
) -> float:
    """Per-class Dice of one predicted label against the case's truth,
    restricted to supervised voxels — the core of `standalone._dice_rows`,
    kept local so the vanilla tree never imports its autonomous entry.

    An unannotated voxel is unknown, not background: where the case carries a
    mask, only voxels the mask marks labelled count in either the prediction
    or the truth. A class absent from BOTH sides has no evidence and scores
    0.0 exactly as the evaluator scores it (the `denom else 0.0` branch).
    """
    supervised = (
        np.ones(case.label.shape, dtype=bool)
        if case.mask is None
        else case.mask.max(axis=0) > 0
    )
    per_class = []
    for cls in range(num_classes):
        p = (predicted == cls) & supervised
        t = (case.label == cls) & supervised
        denom = int(p.sum()) + int(t.sum())
        per_class.append(2.0 * float((p & t).sum()) / denom if denom else 0.0)
    return float(np.mean(per_class[1:])) if num_classes > 1 else 0.0
