# =====================================================================================
# medos-trainer -- the image that actually touches the pixels.
#
# READ THIS FILE TO FIND OUT WHAT FITTED THE MODEL. That is its job. Everything the
# platform records about a training run under `MOS-TRAIN-124` describes software, and
# this Dockerfile plus `requirements.txt` beside it are the whole of that software.
#
# WHY THERE ARE TWO IMAGES AND NOT ONE
# -------------------------------------
# `medicalos/medos` -- the API, the worker, the gateway -- carries NO torch, NO MONAI and
# NO nnU-Net, and that is the best property the training code has. `medos/medos/training/
# chain.py` generates MONAI Bundle configs as DATA (`{"_target_": "monai.transforms.
# Flip"}`) and guards against a transform resolving to `monai.transforms` instead of the
# MedicalOS one; `MOS-TRAIN-225` forbids the nnU-Net planner and `nnUNetPlansManager`
# from the serving image's import closure by name; `MOS-REL-108` forbids in-process
# plugin loading. A single image would make all three unenforceable by construction.
#
# So the platform starts this image's entrypoint and talks to it over FILES AND EXIT
# CODES -- `medos/medos/training/orchestrator.py` documents the run directory and
# `medos_trainer/contract.py` implements the other side of it. There is no socket, no
# RPC and no shared process.
#
# WHY THE BASE IS THE PLATFORM'S PYTHON AND THE CUDA USERSPACE COMES FROM pip
# ----------------------------------------------------------------------------
# The obvious base is `nvidia/cuda:12.4.x-base-ubuntu22.04`. It is not used, and the
# reason is pinning rather than size. Ubuntu 22.04 ships Python 3.10; CONTRACT.md §11
# and `pyproject.toml` require 3.11, and `medos/medos/training/fixtures.py` records two
# byte-exact hashes under "numpy 1.26.4, CPython 3.11" that `MOS-TRAIN-133` compares
# against -- so a 3.10 interpreter here would make the golden fixture a different
# fixture. Getting 3.11 onto jammy means a third-party apt repository, which is a second
# unpinned supply chain in an image whose entire purpose is to be readable.
#
# Starting from the platform's own interpreter pin instead gives:
#   * the SAME CPython the platform runs, down to the patch, so the spec fixtures and the
#     bundle writer behave identically on both sides of the boundary;
#   * every CUDA library pinned by ONE resolver -- `torch==2.5.1+cu124` declares
#     `nvidia-cudnn-cu12` and friends as pinned wheels, so `pip freeze` is the complete
#     inventory. An `nvidia/cuda` base plus pip torch installs the CUDA userspace TWICE,
#     from two pinning systems, and leaves the dynamic loader to pick.
# The driver is NOT in the image either way; it is injected by the container runtime, and
# `NVIDIA_DRIVER_CAPABILITIES` below is what makes `nvidia-smi` and the compute stack
# appear when compose grants the device.
#
# THE BUILD STAMP, AND REGISTER ENTRY 82
# ---------------------------------------
# `MEDOS_TRAINING_ENVIRONMENT` needs nine keys. Seven are deployment properties. Two --
# `code_commit` and `image_digest` -- are BUILD facts, and entry 82 records what happens
# when they are typed into a compose file: "correct exactly until the next
# `docker compose build` and silently false afterwards -- and it is silently false in the
# provenance record of every training run". `MOS-TRAIN-126` requires them "recorded
# rather than asserted".
#
# So they are stamped HERE, at build time, into `/opt/medos-trainer/build-stamp.json`,
# and the entrypoint reads them back. Nobody types them and nobody can: the build REFUSES
# without a commit (see the guard below), and `image_digest` is computed by the image
# about itself over its own installed content. `trainer/build.sh` is the build.
#
# Spec: MOS-TRAIN-121 C1, MOS-TRAIN-124, MOS-TRAIN-126, MOS-TRAIN-135, MOS-TRAIN-190,
# MOS-TRAIN-223, MOS-TRAIN-225, MOS-REL-037, MOS-REL-039, MOS-REL-108,
# docs/spec/99-known-inconsistencies.md entry 82.
#
# BUILD (never `docker build` by hand -- the stamp comes from git):
#   trainer/build.sh
# =====================================================================================
FROM python:3.11.16-slim-bookworm

# What the container runtime needs in order to hand this image a GPU. Neither variable
# puts a driver in the image: they tell the NVIDIA container runtime which devices to
# expose and which driver capabilities to inject. `utility` is what brings `nvidia-smi`,
# which `medos_trainer.environment` reads the driver version out of; `compute` is CUDA.
# Without `--gpus` / `deploy.resources.reservations.devices` on the service these are
# inert, and the entrypoint REFUSES rather than falling back to CPU -- see
# `medos_trainer/environment.py::observe_hardware`.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1: torch's CPU kernels link OpenMP.
#
# gcc/g++: NOT optional, MEASURED. nnU-Net 2.5.1 wraps its network in `torch.compile`
# unless `nnUNet_compile=f`, and TorchInductor generates C++ for the CPU-side wrapper at
# the first forward pass. Without a compiler the fit dies in epoch 0 with
# `BackendCompilerFailed: Failed to find C compiler` -- AFTER the fingerprint has been
# frozen and the run has already left PENDING, which is the most expensive place to
# discover a missing 60 MB package. The alternative, `nnUNet_compile=f`, trades a
# permanent throughput loss on every full run for an image that is smaller than its own
# CUDA wheels; that is the wrong trade for the one image in the stack whose job is
# compute. Setting `nnUNet_compile=f` remains available as deployment configuration.
#
# git: NOT installed, deliberately -- the commit is a build argument and an image that
# can run `git` is an image that can be tempted to re-derive one at run time from a tree
# nobody pinned.
#
# `Acquire::Retries`: the Debian mirror returned a 500 mid-body on `libasan8` on the
# first attempt here. A build that fails on one flaky read is a build people learn to
# re-run without reading, which is how a real failure gets re-run too.
RUN apt-get update -o Acquire::Retries=8 \
 && apt-get install -y --no-install-recommends -o Acquire::Retries=8 libgomp1 gcc g++ \
 && rm -rf /var/lib/apt/lists/*

# ---- the training toolchain -------------------------------------------------------- #
# One layer, pinned by `requirements.txt`. This is the layer a reviewer diffs.
COPY trainer/requirements.txt /tmp/requirements.txt
# The CUDA wheels are large -- `nvidia_cudnn_cu12` alone is 665 MB -- and pip's default
# 15-second socket timeout fails the whole resolve on one slow read. Measured on this
# build, not anticipated: the first attempt died on `nvidia-cusolver-cu12` with
# `TimeoutError: The read operation timed out` after 266 seconds of successful
# downloading. The BuildKit cache mount is what makes the retry cheap, and it is a mount
# rather than a layer, so the image does not carry 6 GB of wheels.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 120 --retries 10 -r /tmp/requirements.txt

# ---- the platform package, WITHOUT its dependencies -------------------------------- #
# `--no-deps` is load-bearing in both directions. Forwards: the trainer must not pull
# fastapi, uvicorn, argon2 or psycopg's binary libpq into an image that serves nothing.
# Backwards: it means `requirements.txt` above must already satisfy every import the
# trainer makes of `medos`, so a drift between the two images fails at BUILD time with an
# ImportError rather than at run time with two numpys.
#
# CORRECTED. This comment used to read: "What the trainer imports from `medos` is narrow
# and pure -- `medos.core.canonical`, `medos.training.{spec,chain,preprocess,bundle,
# autoconfig,errors,fixtures}` -- none of which touches the database, the network or
# FastAPI." That sentence is true of ONE OF THE TWO PROGRAMS in this image and false of
# the image. It omitted `medos.training.{runs,cohort,retrieval,orchestrator}` -- which
# ARE the database and the job queue -- and it listed `chain`, which no file under
# `trainer/` imports by name (it arrives transitively, through `bundle`).
#
# WHAT IS ACTUALLY IN THE IMAGE, measured by AST closure and asserted by
# `tests/unit/test_trainer_import_boundary.py`:
#
#   THE FITTER    plan / fit / declare-environment / doctor
#                 9 modules, 4 013 lines: canonical, spec, preprocess, bundle, fixtures,
#                 autoconfig, chain, training.errors, evidence.errors.
#                 Zero psycopg, zero `medos.db`, zero FastAPI, zero `requests`. The old
#                 sentence's claim, and it holds -- of this half.
#
#   THE SUPERVISOR  execute / execute --watch, which is what this image RUNS
#                   (`docker-compose.yml`: command ["execute", "--watch", "15"])
#                   26 modules, 11 055 lines: `medos.db.{tenancy,audit}`,
#                   `medos.evidence.{leakage,manifest}`, `medos.dicomweb.{client,gateway}`
#                   and five `medos.capabilities.*` modules.
#
# The two closures share exactly three modules -- the digest rule and two refusal types.
#
# SO `COPY medos` IS NOT A NARROW DEPENDENCY. It brings 173 modules / 74 944 lines into
# the image, of which 32 are reachable. `--no-deps` below keeps the image from installing
# fastapi, uvicorn, argon2 and libpq, and that is still load-bearing; what it does not do
# is make the dependency narrow. Narrowing it means moving the supervisor to the platform
# side, where its database already is.
#
# The trainer imports rather than reimplements because a trainer that wrote its own bundle
# writer would be a second implementation of `MOS-TRAIN-129`'s layout, and the platform
# would then be reading a format nobody checked it still emits. That argument survives the
# correction unchanged -- and where those shared modules live is now the SDK inside this
# very distribution: `medos.sdk`, the home `MOS-IMG-003`'s contracts were moved into when
# the platform became the SDK.
COPY pyproject.toml /app/pyproject.toml
# The package INCLUDING THE SDK: `medos/medos/sdk/` is where `MOS-IMG-003`'s contracts
# live now (`medos.sdk`), carried by the one COPY of `medos/medos`. The fitter reaches
# it the way the platform does -- `from medos.sdk import ...` -- which is what lets this
# image install with `--no-deps` and its own pins while both sides still digest through
# the one canonicaliser.
COPY medos/medos /app/medos
WORKDIR /app
RUN pip install --no-cache-dir --no-deps -e .

# ---- the trainer itself ------------------------------------------------------------ #
COPY trainer/medos_trainer /opt/medos-trainer/medos_trainer
COPY trainer/preprocessing-bindings.json /etc/medos/preprocessing-bindings.json
ENV PYTHONPATH=/opt/medos-trainer

# `spec.argv[0]`. A vector, never a shell string (`TrainingRunSpec.__post_init__`), and a
# fixed absolute path rather than `sys.executable` so that the run's `submission_digest`
# does not change with whichever interpreter happened to construct it.
RUN printf '#!/bin/sh\nexec python -m medos_trainer "$@"\n' > /usr/local/bin/medos-trainer \
 && chmod 0755 /usr/local/bin/medos-trainer

# ---- the build stamp: entry 82's two keys, recorded rather than asserted ------------ #
# `build.sh` reads these out of git. The guard below is why a hand `docker build` cannot
# produce an image that lies about its provenance: it fails, loudly, naming the script.
ARG MEDOS_CODE_COMMIT=""
ARG MEDOS_CODE_DIRTY=""
RUN test -n "${MEDOS_CODE_COMMIT}" -a -n "${MEDOS_CODE_DIRTY}" || ( \
      echo "REFUSED: MEDOS_CODE_COMMIT and MEDOS_CODE_DIRTY are build arguments and" >&2; \
      echo "this image will not build without them. MOS-TRAIN-126 requires code_commit" >&2; \
      echo "to be recorded rather than asserted, and register entry 82 records what a" >&2; \
      echo "default does to it. Build with trainer/build.sh." >&2; \
      exit 1 )
RUN python -m medos_trainer.stamp \
      --code-commit "${MEDOS_CODE_COMMIT}" \
      --code-dirty "${MEDOS_CODE_DIRTY}" \
      --out /opt/medos-trainer/build-stamp.json \
 && cat /opt/medos-trainer/build-stamp.json

# Cross-referenceable from `docker inspect` without running the image. The content digest
# is NOT a label: it is computed during the build from the installed set, and a label
# would have to be a second, hand-kept copy of it.
LABEL org.opencontainers.image.title="medos-trainer" \
      org.opencontainers.image.revision="${MEDOS_CODE_COMMIT}" \
      dev.medicalos.trainer.code-dirty="${MEDOS_CODE_DIRTY}" \
      dev.medicalos.trainer.backend="nnunet"

# MOS-TRAIN-121 C1: a separate deployable with a separate service account. Non-root, and
# the same uid the platform image uses so a shared volume needs no chown dance.
RUN useradd --create-home --uid 10001 medos \
 && mkdir -p /var/lib/medos-trainer /run/medos-training \
 && chown -R medos:medos /var/lib/medos-trainer /run/medos-training /opt/medos-trainer
USER medos
ENV nnUNet_raw=/var/lib/medos-trainer/raw \
    nnUNet_preprocessed=/var/lib/medos-trainer/preprocessed \
    nnUNet_results=/var/lib/medos-trainer/results \
    MEDOS_TRAINER_STAMP=/opt/medos-trainer/build-stamp.json

# No ENTRYPOINT with a fixed subcommand: compose gives the environment-stamp service its
# own `command`, and `spec.argv` names `/usr/local/bin/medos-trainer` explicitly.
ENTRYPOINT ["/usr/local/bin/medos-trainer"]
CMD ["--help"]
