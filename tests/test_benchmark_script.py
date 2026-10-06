# SPDX-License-Identifier: Apache-2.0
"""The PulmoAI benchmark example: source-gate style, no real data.

The benchmark script (`examples/benchmark_pulmo.py`) converts 20 real CT cases
into both frameworks' formats — too heavy for CI. So this gate checks what
must stay true without the data:

  * the script exists and its ``--help`` parses (run via subprocess, exactly
    the way a confused user would call it);
  * ``assign_split`` — the split the published numbers rest on — is a pure
    deterministic function of (case_ids, seed): same seed, same assignment;
    every case assigned exactly once; the union covers the input.
  * conversion round-trips on a HANDFUL of synthetic NRRD cases the test
    writes itself (skipped when pynrrd is absent — it is not a pinned
    dependency of this tree), producing the npz members and the dataset.json
    shape the real run relies on.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "benchmark_pulmo.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("benchmark_pulmo", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_exists() -> None:
    assert EXAMPLE.is_file(), f"{EXAMPLE} missing — the benchmark has no converter"


def test_help_parses_via_subprocess() -> None:
    result = subprocess.run(
        [sys.executable, str(EXAMPLE), "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, (
        f"--help exited {result.returncode}: {result.stderr}"
    )
    assert "--convert-only" in result.stdout


def test_assign_split_is_deterministic() -> None:
    script = _load_script()
    ids = [str(i) for i in range(1, 21)]
    first = script.assign_split(ids, seed=0)
    second = script.assign_split(ids, seed=0)
    assert first == second
    assert script.assign_split(list(reversed(ids)), seed=0) == first, (
        "assignment must not depend on the input's order"
    )


def test_assign_split_covers_disjointly() -> None:
    script = _load_script()
    ids = [str(i) for i in range(1, 21)]
    assignment = script.assign_split(ids, seed=0)
    assert set(assignment) == set(ids)
    assert sorted(assignment.values()) == ["train"] * 16 + ["val"] * 4
    for bad in ([], ["1", "1"]):
        with pytest.raises(ValueError):
            script.assign_split(bad, seed=0)


def test_assign_split_respects_the_seed() -> None:
    script = _load_script()
    ids = [str(i) for i in range(1, 21)]
    zero = script.assign_split(ids, seed=0)
    one = script.assign_split(ids, seed=1)
    assert zero != one, "different seeds must give different assignments"
    assert sorted(c for c, side in zero.items() if side == "val") != sorted(
        c for c, side in one.items() if side == "val"
    )


def _write_nrrd_pair(root: Path, case_id: str, shape: tuple[int, int, int]) -> None:
    nrrd = pytest.importorskip("nrrd")
    case_dir = root / case_id
    case_dir.mkdir(parents=True)
    header = {
        "type": "int32",
        "encoding": "raw",
        "space directions": np.eye(3).tolist(),
        "sizes": list(shape),
    }
    nrrd.write(str(case_dir / f"{case_id}.nrrd"),
               np.zeros(shape, dtype=np.int32), header)
    seg = np.zeros(shape, dtype=np.uint8)
    seg[2:4, 2:4, 2:4] = 1
    nrrd.write(str(case_dir / f"{case_id}.seg.nrrd"), seg, dict(header, type="uint8"))


def test_conversion_round_trips_on_synthetic_cases(tmp_path) -> None:
    pytest.importorskip("nrrd")
    pytest.importorskip("nibabel")
    script = _load_script()
    src = tmp_path / "src"
    for case_id in ("1", "2", "3", "4", "5"):
        _write_nrrd_pair(src, case_id, (8, 8, 10))
    out = tmp_path / "bench"

    script.convert(src, out, limit=5, seed=0)

    split = json.loads((out / "split.json").read_text(encoding="utf-8"))
    assert split["seed"] == 0
    assert len(split["train"]) + len(split["val"]) == 5
    assert json.loads((out / "nnunet_raw" / script.DATASET_NAME / "dataset.json").read_text(
        encoding="utf-8"))["numTraining"] == len(split["train"])
    for side in ("train", "val"):
        cases = sorted((out / f"npz_{side}").glob("*.npz"))
        assert cases, f"no npz cases written for {side}"
        with np.load(cases[0]) as z:
            assert z["image"].dtype == np.float32 and z["image"].ndim == 4
            assert z["label"].dtype == np.int64
            assert set(np.unique(z["label"])).issubset({0, 1})
            assert len(z["spacing_mm"]) == 3
    # the nnU-Net layout carries the TRAIN cases only — val stays held out
    images_tr = sorted((out / "nnunet_raw" / script.DATASET_NAME / "imagesTr").glob("*.nii.gz"))
    assert len(images_tr) == len(split["train"])
    # LAZY-IMPORT HONESTY: the whole script module was exec'd by _load_script
    # above, so a module-level `import nrrd` would have bound the name in its
    # namespace. It must not — nrrd is paid for only inside convert(). (A
    # runtime sys.modules check is meaningless here: this test process holds
    # nrrd from the fixture that writes the synthetic NRRD files.)
    assert "nrrd" not in vars(script), (
        "benchmark_pulmo.py binds nrrd at module level — the lazy import is gone"
    )
