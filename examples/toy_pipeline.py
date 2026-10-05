# SPDX-License-Identifier: Apache-2.0
"""End-to-end toy: data -> fingerprint -> plan -> fit -> predict, no platform.

Run it:

    python trainer/examples/toy_pipeline.py

It writes a toy corpus into a temp directory, trains the vanilla UNet on it
for a handful of epochs (CPU, under a minute), and runs the sliding-window
predictor over a held-out case — the same path `vanilla-fit` drives, minus
the CLI. Read it after `medos_trainer/vanilla/`; it is the pipeline the
modules build, in one page.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_toy_corpus(out: Path, n: int = 6, shape: tuple[int, int, int] = (24, 24, 24)) -> None:
    """Bright sphere = class 1, background -600 HU-ish; a corner is unlabelled."""
    rng = np.random.default_rng(0)
    for i in range(n):
        image = rng.normal(-600.0, 50.0, (1, *shape)).astype(np.float32)
        label = np.zeros(shape, dtype=np.int64)
        c = rng.integers(8, 16, size=3)
        kk, jj, ii = np.ogrid[: shape[0], : shape[1], : shape[2]]
        ball = (kk - c[0]) ** 2 + (jj - c[1]) ** 2 + (ii - c[2]) ** 2 <= 5**2
        image[0][ball] = 200.0
        label[ball] = 1
        mask = np.ones_like(image)
        mask[:, : shape[0] // 2, : shape[1] // 2, : shape[2] // 2] = 0.0
        np.savez(out / f"toy-{i}.npz", image=image, label=label, mask=mask)


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="medlange-toy-"))
    cases = work / "cases"
    cases.mkdir()
    make_toy_corpus(cases)

    from medos_trainer.standalone import fit_command, plan_command
    from medos_trainer.vanilla.data import load_case_npz
    from medos_trainer.vanilla.infer import load_predictor

    plan = plan_command(cases, "cpu", work / "plan.json")
    print("plan:", json.dumps({k: plan[k] for k in ("patch_size", "stem_stride", "preset")}))
    for reason in plan["reasons"]:
        print(f"  because {reason}")

    summary = fit_command(cases, "cpu", work / "bundle",
                          epochs=6, steps_per_epoch=8, seed=0)
    print("fit:", json.dumps(summary["best_val_masked_dice_loss"]), "best val masked dice")

    predictor = load_predictor(work / "bundle")
    case = load_case_npz(sorted(cases.glob("*.npz"))[-1])
    label, _ = predictor.predict(case.image)
    fg = (case.label == 1) & (case.mask[0] > 0)
    dice = 2.0 * float((label == 1)[fg].sum()) / float((label == 1).sum() + fg.sum())
    print(f"held-out dice: {dice:.3f} (work dir: {work})")


if __name__ == "__main__":
    main()
