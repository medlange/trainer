# SPDX-License-Identifier: Apache-2.0
"""Cascades: the coarse model's probability map becomes the fine model's input.

THE CASCADE IDEA, in one sentence: a first-stage (coarse) network segments
the whole volume at whatever resolution it was planned for, and a
second-stage (fine) network re-segments the same volume WITH the coarse
prediction as an extra input channel, so the fine stage spends its capacity
on refinement instead of rediscovering the anatomy.

WHAT THIS MODULE OWNS is exactly the hand-off: `build_cascade_cases` runs
the coarse predictor over each case and appends the coarse FOREGROUND
probability as one channel. For a binary coarse model that is `probs[1]`,
the foreground class; for a multi-class coarse model it is the MAX over
classes 1..C-1 — one "any-structure" guide channel either way, which is what
a single appended channel can honestly carry. The fine case keeps the
original label, spacing and case_id; its mask gains a matching all-ones
channel, because the guide channel is COMPUTED, hence supervised, everywhere.
"""

from __future__ import annotations

import numpy as np
from medos_trainer.vanilla.data import Case
from medos_trainer.vanilla.infer import SlidingWindowPredictor


def build_cascade_cases(
    coarse_predictor: SlidingWindowPredictor,
    cases: list[Case],
    device: str = "cpu",
) -> list[Case]:
    """Coarse inference over each case -> `Case`s with one extra image channel.

    The appended channel is the coarse foreground probability: `probs[1]` for
    a binary coarse model, `probs[1:].max(axis=0)` for a multi-class one.
    Everything else about the case is untouched — same label, same spacing,
    same case_id — and the mask is extended with an all-ones channel so it
    keeps matching the image's (C+1, K, J, I) shape.

    `device` names where the CASCADE will run; the coarse predictor itself
    already knows the device it serves on (`load_predictor(checkpoint_dir,
    device=...)`), so the value is recorded intent rather than a mutation —
    it keeps the signature honest for callers that build the predictor in the
    same breath.
    """
    del device  # the coarse predictor carries its own serving device
    num_classes = int(coarse_predictor.net.config.num_classes)
    out = []
    for case in cases:
        _, probs = coarse_predictor.predict(case.image)
        guide = probs[1] if num_classes == 2 else probs[1:].max(axis=0)
        image = np.concatenate(
            [case.image, guide.astype(np.float32)[None]], axis=0
        )
        if case.mask is None:
            mask = None
        else:
            # THE GUIDE CHANNEL IS COMPUTED, NOT ANNOTATED — it is supervised
            # at every voxel, which is exactly what an all-ones channel says.
            extra = np.ones((1,) + case.label.shape, dtype=case.mask.dtype)
            mask = np.concatenate([case.mask, extra], axis=0)
        out.append(Case(image=image, label=case.label, mask=mask,
                        spacing_mm=case.spacing_mm, case_id=case.case_id))
    return out
