# SPDX-License-Identifier: Apache-2.0
"""The build stamp: `code_commit`, `code_dirty` and `image_digest`, recorded at build.

REGISTER ENTRY 82, WHICH THIS MODULE EXISTS TO CLOSE
-----------------------------------------------------
`MEDOS_TRAINING_ENVIRONMENT` needs nine keys and `medos/medos/api/routes_training.py` answers
`503 TRAINING_ENVIRONMENT_NOT_RECORDED` without them. Seven describe the deployment and a
static declaration keeps them true. Two do not, and entry 82 says exactly why:

    "`code_commit` and `image_digest` ... change with every build, so a value written into
    `docker-compose.yml` is correct exactly until the next `docker compose build` and
    silently false afterwards -- and it is silently false in the provenance record of
    every training run, which is the one place `MOS-TRAIN-126` exists to make
    trustworthy."

`MOS-TRAIN-126`'s own words are that these are "recorded rather than asserted", and
`medos.training.runs.RunBinding` carries no defaults for the same reason: "a default is an
assertion wearing a record's clothes". So this module runs ONCE, inside the image, during
`docker build`, and writes what it OBSERVED to a file the entrypoint reads back. After a
rebuild the file is different because the image is different. Nobody edits anything.

WHAT `image_digest` IS HERE, AND WHY IT IS NOT THE OCI MANIFEST DIGEST
-----------------------------------------------------------------------
It is `sha256:` over the canonical JSON of an inventory the image takes of itself: the
interpreter, the platform, the resolved installed Python distributions, the dpkg package
set, and the content hash of every source file the build copied in. Three reasons for
that choice over the registry digest, and the third is the one that decides it:

  1. AVAILABILITY. A container cannot read its own OCI manifest digest without a route to
     the daemon, and giving the trainer a docker socket to look up its own provenance is a
     larger hole than the one it closes (`MOS-REL-039`; chapter 15's "zero orchestrator API
     calls from application code").
  2. THE VALUE WOULD BE CIRCULAR. Writing the final image's digest into the final image
     changes the final image.
  3. AN OCI IMAGE ID IS NOT REPRODUCIBLE AND THIS FIELD MUST BE. `MOS-TRAIN-124`'s
     `run_digest` is a digest over the whole binding INCLUDING `image_digest`, and
     `training_runs_run_digest_uk` makes two runs with the same digest the same
     experiment. A docker image id changes on every `docker build` -- it carries layer
     creation timestamps -- so recording it would make the same experiment, rebuilt from
     the same commit with the same pins, look like a different experiment. An inventory
     digest changes when and only when the software changes, which is the property the
     field is used for.

This is a deliberate reading of `MOS-TRAIN-124`'s "container image digest" and it is
REPORTED as such rather than smuggled: the registry digest and this digest answer
different questions, and the one the binding needs is "what software was this", not "what
blob did the registry store". `trainer/build.sh` prints the OCI image id beside
this value at the end of every build so the two can be correlated by hand.

Stdlib only: this runs during `docker build`, before anything is guaranteed importable
except what `requirements.txt` installed, and a stamp that needed a dependency to compute
itself would be a stamp that can fail for a reason unrelated to the image.

Spec: MOS-TRAIN-124, MOS-TRAIN-125, MOS-TRAIN-126, MOS-REL-037,
docs/spec/99-known-inconsistencies.md entry 82.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess  # noqa: S404 - `pip freeze`, argv vector, no shell
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Final

__all__ = [
    "STAMP_KEYS",
    "build_stamp",
    "content_digest",
    "main",
    "read_stamp",
]

#: `medos.training.runs._COMMIT_RE`, restated so a bad commit is refused at BUILD rather
#: than at the first submit. The platform accepts a 40- or 64-hex object name.
_COMMIT_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")

#: What the stamp file carries. Exactly entry 82's two keys, plus the inventory the digest
#: was taken over -- kept beside the digest so a reviewer can recompute it rather than
#: trust it.
STAMP_KEYS: Final[tuple[str, ...]] = ("code_commit", "code_dirty", "image_digest")

#: The trees whose content the image digest covers. Everything the build COPYied in, and
#: nothing that a run can write: `/var/lib/medos-trainer` is scratch and is excluded, or
#: the digest would change after the first training run.
_HASHED_TREES: Final[tuple[str, ...]] = (
    "/app/medos",
    "/app/pyproject.toml",
    "/opt/medos-trainer/medos_trainer",
    "/tmp/requirements.txt",  # noqa: S108 - the build's own copy, not a run-time path
)


def _canonical(document: Mapping[str, Any]) -> bytes:
    """JCS-shaped bytes: sorted keys, no spaces, UTF-8. `MOS-EVID-008`'s form.

    `medos.sdk.canonical.canonical_bytes` is the ONE implementation -- shared by the
    platform and this image rather than owned by either (`MOS-REL-032`) -- and is what
    every other digest in this tree goes through. It is not imported here because this
    module runs during the build with stdlib only; the two agree on the two properties
    that matter for a digest -- key order and separator -- and
    `tests/unit/test_trainer_stamp.py` asserts that agreement against the real one.
    """
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hashes(roots: Iterable[str]) -> dict[str, str]:
    """`{path: sha256}` for every file under `roots`, sorted, `__pycache__` excluded.

    Compiled bytecode is excluded because it is not source: its content depends on the
    interpreter's own hash seed and on when it was written, and including it would make
    the digest change for a reason nobody made.
    """
    out: dict[str, str] = {}
    for root in roots:
        base = Path(root)
        if base.is_file():
            out[str(base)] = _sha256_file(base)
            continue
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
                continue
            out[str(path).replace("\\", "/")] = _sha256_file(path)
    return dict(sorted(out.items()))


def _installed_distributions() -> dict[str, str]:
    """`{name: version}` for everything pip resolved. The real inventory, not the pins.

    `requirements.txt` states what was ASKED for; this states what arrived, including
    every transitive wheel torch dragged in -- `nvidia-cudnn-cu12` among them. That is the
    set `MOS-TRAIN-126`'s `framework_versions` is read from and the set a reviewer needs in
    order to answer "what touched the data".
    """
    try:
        from importlib.metadata import distributions
    except ImportError:  # pragma: no cover - stdlib since 3.8
        return {}
    out: dict[str, str] = {}
    for dist in distributions():
        name = (dist.metadata["Name"] or "").strip()
        if name:
            out[name.lower()] = str(dist.version)
    return dict(sorted(out.items()))


def _os_packages() -> dict[str, str]:
    """`{package: version}` from dpkg, or `{}` off Debian.

    MEASURED GAP, CLOSED. The first version of this inventory covered only the Python
    distributions and the copied source, so adding `gcc` and `g++` to the image -- which
    is what makes `torch.compile` work at all -- produced a BYTE-IDENTICAL
    `image_digest` for a materially different image. That is the entry-82 defect in a
    new place: a provenance field that does not move when the thing it describes does.
    """
    binary = shutil.which("dpkg-query")
    if binary is None:
        return {}
    try:
        result = subprocess.run(  # noqa: S603 - argv vector, no shell
            [binary, "-W", "-f", "${Package} ${Version}\n"],
            capture_output=True, text=True, timeout=120, check=False,
        )
    except OSError:  # pragma: no cover - dpkg present and unrunnable is not a case
        return {}
    if result.returncode != 0:
        return {}
    out: dict[str, str] = {}
    for line in result.stdout.splitlines():
        name, _, version = line.partition(" ")
        if name:
            out[name] = version.strip()
    return dict(sorted(out.items()))


def content_digest(*, trees: Iterable[str] = _HASHED_TREES) -> tuple[str, dict[str, Any]]:
    """`(sha256:..., inventory)` -- what this image IS, computed by this image.

    Returns the inventory beside the digest so the stamp can carry both and the digest
    stays recomputable from the file alone. See the module docstring for why this and not
    the OCI manifest digest.
    """
    inventory: dict[str, Any] = {
        "interpreter": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "abiflags": getattr(sys, "abiflags", ""),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "libc": "-".join(x for x in platform.libc_ver() if x),
        },
        "distributions": _installed_distributions(),
        "os_packages": _os_packages(),
        "files": _tree_hashes(trees),
    }
    return "sha256:" + hashlib.sha256(_canonical(inventory)).hexdigest(), inventory


def build_stamp(*, code_commit: str, code_dirty: bool) -> dict[str, Any]:
    """The document written into the image. Refuses a commit that is not one.

    `code_dirty` is NOT refused here and must not be. `MOS-TRAIN-125` blocks
    `state = SUCCEEDED` for a dirty tree, which is a different statement and belongs to
    `medos.training.runs.succeed`: "a dirty-tree run is a legitimate experiment and an
    illegitimate candidate, and refusing it at submit would push people to commit noise in
    order to run anything." Refusing it at BUILD would be worse still -- it would stop a
    developer building an image to debug with.
    """
    commit = code_commit.strip().lower()
    if not _COMMIT_RE.match(commit):
        raise ValueError(
            f"code_commit {code_commit!r} is not a git object name. MOS-TRAIN-125 pins "
            "the commit a run was fitted from and medos.training.runs refuses anything "
            "that is not 40 or 64 lowercase hex characters; a placeholder here would be "
            "recorded in the provenance of every run this image performs"
        )
    digest, inventory = content_digest()
    return {
        "code_commit": commit,
        "code_dirty": bool(code_dirty),
        "image_digest": digest,
        # Kept so the digest above is recomputable from this file rather than trusted.
        "image_inventory": inventory,
    }


def read_stamp(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Read the stamp the build wrote. Raises `FileNotFoundError`; never defaults.

    An unstamped image has no answer to entry 82 and must say so, loudly, at the point the
    environment is declared -- not quietly, with a zero commit, in the binding of a run
    somebody will later try to reproduce.
    """
    target = Path(path or os.environ.get("MEDOS_TRAINER_STAMP", "")
                  or "/opt/medos-trainer/build-stamp.json")
    if not target.is_file():
        raise FileNotFoundError(
            f"no build stamp at {target}. This image was not built by "
            "trainer/build.sh, so it cannot say which commit or which content it "
            "carries, and MOS-TRAIN-126 requires both to be recorded rather than "
            "asserted (register entry 82)"
        )
    document = json.loads(target.read_text(encoding="utf-8"))
    missing = [k for k in STAMP_KEYS if k not in document]
    if missing:
        raise ValueError(f"the build stamp at {target} is missing {missing}")
    return dict(document)


def _dirty_flag(raw: str) -> bool:
    text = raw.strip().lower()
    if text in ("1", "true", "yes", "dirty"):
        return True
    if text in ("0", "false", "no", "clean"):
        return False
    raise ValueError(
        f"--code-dirty {raw!r} is neither true nor false. It is a recorded fact about "
        "the tree the image was built from and there is no third value"
    )


def git_facts(repo: str | os.PathLike[str] = ".") -> tuple[str, bool]:
    """`(commit, dirty)` for a working tree. Used by `build.sh`, never by the image.

    Lives here rather than in the shell script so that the definition of "dirty" is one
    definition: `git status --porcelain` over tracked AND untracked files, because an
    untracked module that got COPYied into the image is exactly the change a re-run would
    not reproduce.
    """
    root = str(repo)
    commit = subprocess.run(  # noqa: S603 - argv vector, no shell
        ["git", "-C", root, "rev-parse", "HEAD"],  # noqa: S607
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    status = subprocess.run(  # noqa: S603 - argv vector, no shell
        ["git", "-C", root, "status", "--porcelain"],  # noqa: S607
        capture_output=True, text=True, check=True,
    ).stdout
    return commit, bool(status.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="medos_trainer.stamp",
        description="Write the build stamp into the image. Runs during docker build.",
    )
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--code-dirty", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    document = build_stamp(
        code_commit=args.code_commit, code_dirty=_dirty_flag(args.code_dirty)
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Printed without the inventory: it is thousands of lines and the build log is read.
    print(json.dumps({k: document[k] for k in STAMP_KEYS}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - a process entry point
    raise SystemExit(main())
