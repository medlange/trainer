#!/usr/bin/env sh
# =====================================================================================
# Build the Medlange Trainer image. THIS IS THE BUILD -- `docker build` by
# hand is refused.
#
# Register entry 82: `code_commit` and `image_digest` "change with every
# build, so a value written into `docker-compose.yml` is correct exactly
# until the next `docker compose build` and silently false afterwards". The
# fix is to stamp them at build time, and this script is the half of that
# which has to run outside the image, because the image deliberately cannot
# see the git tree (`Dockerfile.dockerignore` excludes `.git`, and an image
# that can run `git rev-parse` is an image that can report a commit for a
# tree it was not built from).
#
# What it does, and nothing else:
#   1. reads the commit and the dirty flag out of git, through the ONE
#      definition of "dirty" this project has (`medos_trainer.stamp.git_facts`);
#   2. hands them to the build as arguments -- the Dockerfile refuses an
#      empty one;
#   3. prints the stamp the image computed about ITSELF, beside the OCI
#      image id, so the two identifiers can be correlated by hand. See
#      `medos_trainer/stamp.py` for why the recorded `image_digest` is the
#      inventory digest and not the OCI id.
#
# USAGE
#   trainer/build.sh                 # tag medlange/trainer:0.3.0.dev0
#   MEDOS_TRAINER_TAG=x trainer/build.sh
#
# Spec: register entry 82; MOS-REL-037 (the pins are recorded versions).
# =====================================================================================
set -eu

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/.." && pwd)
tag=${MEDOS_TRAINER_TAG:-medlange/trainer:0.3.0.dev0}

# `git_facts` and not two inline `git` calls: the definition of a dirty tree
# is one definition, in `stamp.py`, beside the field it is recorded into.
# THE FACTS COME FROM GIT -- FROM THE TREE THIS CONTEXT IS, OR THE ONE IT
# WAS COPIED FROM.
#
# The image is built where the GPU is, and that is not always the machine
# holding the history: a card this developer's machine does not have means
# the context is copied to that host and built there. `.git` is NOT copied
# with it, and must not be -- `Dockerfile.dockerignore` gives the reason.
#
# So a build outside the history is TOLD which tree it is building, and the
# caller asserts the one thing no tool here can check: that this context is
# a faithful copy of that tree. The assertion is explicit and both halves
# are required. A silent fallback -- to an empty commit, to "unknown", or
# worst of all to `dirty=false` -- would stamp a reproducible-looking flag
# on an image nobody can reproduce. Refusing is the only honest default.
if git -C "$repo" rev-parse --git-dir >/dev/null 2>&1; then
  facts=$(cd "$repo" && python -c '
import json, sys
sys.path.insert(0, "trainer")
from medos_trainer.stamp import git_facts
commit, dirty = git_facts(".")
print(json.dumps({"commit": commit, "dirty": "true" if dirty else "false"}))
')
  commit=$(printf '%s' "$facts" | python -c 'import json,sys; print(json.load(sys.stdin)["commit"])')
  dirty=$(printf '%s' "$facts" | python -c 'import json,sys; print(json.load(sys.stdin)["dirty"])')
  origin=observed
else
  commit=${MEDOS_CODE_COMMIT:-}
  dirty=${MEDOS_CODE_DIRTY:-}
  if [ -z "$commit" ] || [ -z "$dirty" ]; then
    echo "medlange-trainer: $repo holds no git repository, and MEDOS_CODE_COMMIT /" >&2
    echo "  MEDOS_CODE_DIRTY are not both set. A build outside the history has to be told" >&2
    echo "  which tree it is building; set both from the machine that has that tree and" >&2
    echo "  re-run. Nothing is guessed here on purpose -- see the comment above." >&2
    exit 2
  fi
  case "$dirty" in
    true|false) ;;
    *) echo "medlange-trainer: MEDOS_CODE_DIRTY is 'true' or 'false', got '$dirty'" >&2; exit 2 ;;
  esac
  origin=supplied
fi

echo "medlange-trainer: building $tag from commit $commit (dirty=$dirty, facts $origin)"

iidfile=$(mktemp)
trap 'rm -f "$iidfile"' EXIT

# Under Git Bash / MSYS the shell hands `docker` a POSIX path (`/d/...`) that
# the Windows daemon cannot resolve, and the build fails with "unable to
# prepare context". Measured on this deployment, not anticipated. `cygpath`
# is absent everywhere else, so the branch costs nothing on Linux.
context=$repo
dockerfile=$here/Dockerfile
iidpath=$iidfile
if command -v cygpath >/dev/null 2>&1; then
  context=$(cygpath -w "$repo")
  dockerfile=$(cygpath -w "$here/Dockerfile")
  # `--iidfile` is opened by the DAEMON's client on the Windows side, so a
  # POSIX path here fails the whole build after the image has already been
  # exported -- measured, with "writing image ID file: The system cannot find
  # the path specified".
  iidpath=$(cygpath -w "$iidfile")
fi

DOCKER_BUILDKIT=1 docker build \
  --file "$dockerfile" \
  --tag "$tag" \
  --iidfile "$iidpath" \
  --build-arg "MEDOS_CODE_COMMIT=$commit" \
  --build-arg "MEDOS_CODE_DIRTY=$dirty" \
  "$context"

echo
echo "medlange-trainer: built. The two identifiers, which answer different questions:"
echo "  OCI image id (what the daemon stored, not reproducible across builds):"
echo "    $(cat "$iidfile")"
echo "  recorded image_digest (what the software IS, reproducible from commit + pins):"
docker run --rm --entrypoint python "$tag" -c '
import json
from medos_trainer.stamp import read_stamp
s = read_stamp()
print("    " + s["image_digest"])
print("  code_commit: " + s["code_commit"] + "   code_dirty: " + str(s["code_dirty"]))
'
