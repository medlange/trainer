# SPDX-License-Identifier: Apache-2.0
"""`python -m medos_trainer` — the Medlange Trainer command line.

SIX SUBCOMMANDS, AND NOTHING ELSE

    vanilla-plan            fingerprint a directory of `.npz` cases and write the
                            derived plan (patch size, batch size, schedule) with the
                            reasons attached
    vanilla-fit             plan + train + write an inference bundle, no platform:
                            cases in, `model.pt` + `net_config.json` + `fit_plan.json` out
    vanilla-import-nnunet   convert an nnU-Net NIfTI layout (imagesTr/ + labelsTr/)
                            into `.npz` cases the vanilla stack reads
    predict                 run a saved bundle over one case (sliding window)
    declare-environment     emit `MEDOS_TRAINING_ENVIRONMENT`'s nine keys, observed
    doctor                  what this installation can and cannot do right now, as a
                            document — the answer to "is the GPU attached" that does
                            not require starting a training run

THE MODULE IMPORTS NO `medos`. The trainer is a standalone vanilla-PyTorch
framework; every input is a path on the local filesystem and every output is a
file. There is no run directory, no platform handshake and no socket — a
researcher with a folder of volumes reaches a trained model through
`vanilla-fit` alone.

THE TORCH IMPORTS ARE DELIBERATELY INSIDE THE FUNCTIONS that need them, so
`doctor` and `declare-environment` answer on a machine that can import this
package without paying for a torch import they may not need... and so the
module imports cleanly in environments where only the plan/data layers are
wanted.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def _logging() -> None:
    logging.basicConfig(
        level=os.environ.get("MEDOS_TRAINER_LOG_LEVEL", "INFO").upper(),
        format=_LOG_FORMAT,
        stream=sys.stdout,
    )


def _predict(args: argparse.Namespace) -> int:
    """The vanilla stack's standalone inference: bundle in, prediction out.

    EXISTS BECAUSE THE AUDIT NAMED ITS ABSENCE: a trainer a stranger cannot
    run a trained model with is a component, not a framework. No platform,
    no run directory, no database — a checkpoint directory from `fit`, one
    case `.npz`, one output `.npz`.
    """
    import numpy as np
    from medos_trainer.vanilla.data import load_case_npz
    from medos_trainer.vanilla.infer import load_predictor

    case = load_case_npz(args.input)
    predictor = load_predictor(
        args.checkpoint_dir, overlap=args.overlap,
        batch_size=args.batch_size, device=args.device,
    )
    label, probabilities = predictor.predict(case.image)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, label=label, probabilities=probabilities)
    print(f"wrote {out} label={label.shape} probabilities={probabilities.shape}")
    return 0


def _declare_environment(out: str | None, allow_cpu: bool) -> int:
    from medos_trainer.environment import HardwareUnavailable, declare

    try:
        document = declare(allow_cpu=allow_cpu or None)
    except (FileNotFoundError, ValueError, HardwareUnavailable) as exc:
        # REFUSED, NAMED: an unstamped tree or a GPU-less host is a fact about this
        # installation, not a crash. The message names which of the nine keys could
        # not be observed and why.
        print(f"declare-environment: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if out:
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Written whole and then moved, so a reader that opens the file while this is
        # running never sees half a declaration and reports eight of nine keys missing.
        temporary = target.with_suffix(target.suffix + ".partial")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(target)
        target.chmod(0o644)
        print(f"wrote {target}")
    print(text)
    return 0


def _doctor() -> int:
    """What this installation can do, as a document. No training run required."""
    from medos_trainer.environment import HardwareUnavailable, observe_hardware
    from medos_trainer.stamp import read_stamp

    report: dict[str, Any] = {}
    try:
        report["stamp"] = {
            k: v for k, v in read_stamp().items() if k != "image_inventory"
        }
    except (FileNotFoundError, ValueError) as exc:
        report["stamp"] = {"error": str(exc)}
    try:
        report["hardware"] = observe_hardware()
        report["can_fit"] = True
    except HardwareUnavailable as exc:
        report["hardware"] = {"error": str(exc)}
        report["can_fit"] = False
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("can_fit") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medos-trainer",
        description="Medlange Trainer — a vanilla-PyTorch framework for volumetric "
                    "medical image segmentation. No platform, no run directory.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    vanilla_plan = sub.add_parser(
        "vanilla-plan",
        help="fingerprint a cases directory and write the derived plan",
    )
    vanilla_plan.add_argument("--data", required=True, help="directory of .npz cases")
    vanilla_plan.add_argument("--preset", default="cpu")
    vanilla_plan.add_argument("--out", required=True, help="plan JSON path")

    vanilla_fit = sub.add_parser(
        "vanilla-fit",
        help="plan and train over a cases directory, no platform",
    )
    vanilla_fit.add_argument("--data", required=True, help="directory of .npz cases")
    vanilla_fit.add_argument("--preset", default="cpu")
    vanilla_fit.add_argument("--out", required=True, help="bundle directory")
    vanilla_fit.add_argument("--epochs", type=int, default=None,
                           help="override the plan's epochs (smoke runs)")
    vanilla_fit.add_argument("--steps-per-epoch", type=int, default=None)
    vanilla_fit.add_argument("--seed", type=int, default=0)
    vanilla_fit.add_argument("--device", default="cpu")

    vanilla_import = sub.add_parser(
        "vanilla-import-nnunet",
        help="convert an nnU-Net NIfTI layout to .npz cases",
    )
    vanilla_import.add_argument("--images", required=True, help="imagesTr/ directory")
    vanilla_import.add_argument("--labels", required=True, help="labelsTr/ directory")
    vanilla_import.add_argument("--out", required=True, help="output .npz directory")

    declare_parser = sub.add_parser(
        "declare-environment", help="emit MEDOS_TRAINING_ENVIRONMENT's nine keys"
    )
    declare_parser.add_argument("--out", default=None)
    declare_parser.add_argument(
        "--allow-cpu", action="store_true",
        help="record a deployment with no GPU AS one, instead of refusing",
    )

    sub.add_parser("doctor", help="what this installation can and cannot do right now")

    predict_parser = sub.add_parser(
        "predict",
        help="run a saved bundle over one case (sliding window)",
    )
    predict_parser.add_argument(
        "--checkpoint-dir", required=True,
        help="directory fit wrote (model.pt, net_config.json, fit_plan.json)",
    )
    predict_parser.add_argument("--input", required=True, help="case .npz (image)")
    predict_parser.add_argument("--output", required=True, help="output .npz")
    predict_parser.add_argument("--overlap", type=float, default=0.5)
    predict_parser.add_argument("--batch-size", type=int, default=2)
    predict_parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    _logging()
    args = build_parser().parse_args(argv)
    if args.command == "declare-environment":
        return _declare_environment(args.out, args.allow_cpu)
    if args.command == "predict":
        return _predict(args)
    if args.command == "vanilla-plan":
        from medos_trainer.standalone import plan_command

        plan_command(args.data, args.preset, args.out)
        print(f"wrote {args.out}")
        return 0
    if args.command == "vanilla-fit":
        from medos_trainer.standalone import fit_command

        summary = fit_command(
            args.data, args.preset, args.out,
            epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
            seed=args.seed, device=args.device,
        )
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "vanilla-import-nnunet":
        from medos_trainer.standalone import import_nnunet_dataset

        n = import_nnunet_dataset(args.images, args.labels, args.out)
        print(f"converted {n} cases into {args.out}")
        return 0
    return _doctor()


if __name__ == "__main__":  # pragma: no cover - a process entry point
    raise SystemExit(main())
