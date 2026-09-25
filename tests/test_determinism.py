# SPDX-License-Identifier: Apache-2.0
"""The determinism settings that are DECLARED are the ones that get APPLIED.

WHY THIS TEST EXISTS
---------------------
`MOS-TRAIN-126` makes one narrow promise and refuses a wider one:

    "Bit-exact reproducibility of a training run MUST NOT be required and MUST NOT be
    claimed ... What the record above guarantees is weaker and sufficient: every input is
    pinned, THE DETERMINISM SETTINGS ACTUALLY USED ARE RECORDED RATHER THAN ASSERTED, and
    a re-run is a *comparable* run rather than an identical one."

`medos_trainer.environment.DETERMINISM` and `SEEDS` are what the deployment DECLARES in
`MEDOS_TRAINING_ENVIRONMENT`, and they travel into every run's binding through
`medos/api/routes_training.py`. `medos_trainer.backend.apply_determinism` is what the
trainer APPLIES at the start of each phase. If a key is added to the declaration and the
applier never reads it, the binding records a setting nobody set -- which is precisely
the assertion `MOS-TRAIN-126` is written against, and it is invisible: the run succeeds,
the record looks complete, and the flag was never touched.

WHAT IT CHECKS AND WHAT IT CANNOT
-----------------------------------
It reads `backend.py`'s SOURCE rather than calling `apply_determinism`, because calling
it needs torch and this is a unit test that runs in the platform's environment. That is a
real limit and it is stated: this catches a key that is declared and never read, which is
the drift that happens. It does not catch a key that is read and then applied to the
wrong torch switch; `tests/integration/test_trainer_image.py` observes the applied values
coming back in `result.json`, inside the container, which is where that is checkable.

Spec: MOS-TRAIN-124, MOS-TRAIN-126, MOS-TRAIN-127.
"""

from __future__ import annotations

import sys
from pathlib import Path

TRAINER_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINER_ROOT))

from medos_trainer import environment as env  # noqa: E402

_BACKEND_SOURCE = (TRAINER_ROOT / "medos_trainer" / "backend.py").read_text(
    encoding="utf-8"
)


def test_every_declared_determinism_key_is_read_by_the_applier() -> None:
    unread = [k for k in env.DETERMINISM if f'"{k}"' not in _BACKEND_SOURCE]
    assert not unread, (
        f"medos_trainer.environment.DETERMINISM declares {unread}, which "
        "medos_trainer.backend.apply_determinism never reads. MOS-TRAIN-126 requires "
        "the settings ACTUALLY USED to be recorded; a declared-and-unapplied key is a "
        "setting the run's binding claims and the run did not have"
    )


def test_every_declared_seed_is_read_by_the_applier() -> None:
    unread = [k for k in env.SEEDS if f'"{k}"' not in _BACKEND_SOURCE]
    assert not unread, (
        f"medos_trainer.environment.SEEDS declares {unread}, which "
        "medos_trainer.backend.apply_determinism never seeds. MOS-TRAIN-127's "
        "seed-variance study varies a seed; a seed nothing consumes varies nothing"
    )


def test_the_applier_reports_back_rather_than_returning_none() -> None:
    """`apply_determinism` returns what it applied, and `result.json` carries it.

    A function that sets flags and returns nothing gives the record no way to be about
    the application rather than about the intention -- which is the distinction the
    requirement turns on.
    """
    assert "def apply_determinism(request: RunRequest) -> dict[str, Any]:" in _BACKEND_SOURCE
    assert '"applied":' in _BACKEND_SOURCE


def test_the_applier_reads_the_request_and_not_the_declaration_module() -> None:
    """One chain: declaration -> MEDOS_TRAINING_ENVIRONMENT -> binding -> request.json.

    If `backend.py` imported `environment.DETERMINISM` directly it would apply the
    TRAINER's current constants while the run's binding recorded whatever the
    declaration said when the run was submitted -- two values, one name, and a rebuild
    between them.
    """
    assert "from medos_trainer.environment import" not in _BACKEND_SOURCE
    assert "determinism = dict(request.determinism)" in _BACKEND_SOURCE
    assert "seeds = dict(request.seeds)" in _BACKEND_SOURCE
