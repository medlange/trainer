# =====================================================================================
# Medlange Trainer -- the image that actually touches the pixels.
#
# READ THIS FILE TO FIND OUT WHAT FITTED THE MODEL. That is its job: the
# trainer is a vanilla-PyTorch framework (own 3D UNet, own fingerprint-based
# planner, own masked-loss training loop, own sliding-window inference), and
# this Dockerfile plus `requirements.txt` beside it are the whole of the
# software it ships. There is NO nnU-Net and NO MONAI in this image; the
# framework replaces both.
#
# WHAT THE IMAGE RUNS
# --------------------
# `python -m medos_trainer` with six subcommands: `vanilla-plan`,
# `vanilla-fit`, `vanilla-import-nnunet`, `predict`, `declare-environment`
# and `doctor`. Every input is a path on the local filesystem and every
# output is a file -- a cases directory in, an inference bundle out. There
# is no platform handshake, no run directory and no socket.
#
# WHY THE BASE IS A PLAIN PYTHON AND THE CUDA USERSPACE COMES FROM pip
# ---------------------------------------------------------------------
# `torch==2.7.1+cu128` declares `nvidia-cudnn-cu12` and friends as pinned
# wheels, so every CUDA library in this image is pinned by the ONE resolver
# that pinned torch and `pip freeze` is the complete inventory. A CUDA base
# image plus pip torch would install the CUDA userspace TWICE, from two
# pinning systems, and leave the dynamic loader to pick. The driver is NOT
# in the image either way; it is injected by the container runtime, and
# `NVIDIA_DRIVER_CAPABILITIES` below is what makes `nvidia-smi` and the
# compute stack appear when the device is granted.
#
# THE BUILD STAMP, AND REGISTER ENTRY 82
# ---------------------------------------
# `MEDOS_TRAINING_ENVIRONMENT` needs nine keys. Seven are observed from the
# installed software and the granted hardware; two -- `code_commit` and
# `image_digest` -- are BUILD facts, and entry 82 records what happens when
# they are typed into a compose file: "correct exactly until the next
# `docker compose build` and silently false afterwards". The fix is that the
# image computes the two facts about ITSELF at build time. Nobody types them
# and nobody can: the build REFUSES without a commit (see the guard below),
# and `image_digest` is computed by the image about itself over its own
# installed content. `trainer/build.sh` is the build.
#
# BUILD (never `docker build` by hand -- the stamp comes from git):
#   trainer/build.sh
# =====================================================================================
FROM python:3.11.16-slim-bookworm

# What the container runtime needs in order to hand this image a GPU. Neither
# variable puts a driver in the image: they tell the NVIDIA container runtime
# which devices to expose and which driver capabilities to inject. `utility`
# is what brings `nvidia-smi`, which `medos_trainer.environment` reads the
# driver version out of; `compute` is CUDA. Without `--gpus` /
# `deploy.resources.reservations.devices` on the service these are inert, and
# `observe_hardware` REFUSES rather than falling back to CPU -- see
# `medos_trainer/environment.py`.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1: torch's CPU kernels link OpenMP.
#
# git: NOT installed, deliberately -- the commit is a build argument and an
# image that can run `git` is an image that can be tempted to re-derive one
# at run time from a tree nobody pinned.
#
# `Acquire::Retries`: the Debian mirror returned a 500 mid-body on the first
# attempt here. A build that fails on one flaky read is a build people learn
# to re-run without reading, which is how a real failure gets re-run too.
RUN apt-get update -o Acquire::Retries=8 \
 && apt-get install -y --no-install-recommends -o Acquire::Retries=8 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# ---- the training toolchain -------------------------------------------------------- #
# One layer, pinned by `requirements.txt`: torch (with the CUDA userspace as
# its own pinned wheels), numpy, scipy, and nibabel for the NIfTI importer.
# This is the layer a reviewer diffs.
COPY trainer/requirements.txt /tmp/requirements.txt
# The CUDA wheels are large -- `nvidia_cudnn_cu12` alone is 665 MB -- and
# pip's default 15-second socket timeout fails the whole resolve on one slow
# read. Measured on this build, not anticipated: the first attempt died on
# `nvidia-cusolver-cu12` with `TimeoutError: The read operation timed out`
# after 266 seconds of successful downloading. The BuildKit cache mount is
# what makes the retry cheap, and it is a mount rather than a layer, so the
# image does not carry 6 GB of wheels.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 120 --retries 10 -r /tmp/requirements.txt

# ---- the trainer itself ------------------------------------------------------------ #
# The whole product: the framework, the CLI and the stamp. Nothing else is
# copied in -- the trainer imports no platform package, so the image installs
# none.
COPY trainer/medos_trainer /opt/medos-trainer/medos_trainer
ENV PYTHONPATH=/opt/medos-trainer

# The CLI entrypoint. A vector, never a shell string, and a fixed absolute
# path rather than `sys.executable` so the invocation does not change with
# whichever interpreter happened to launch it.
RUN printf '#!/bin/sh\nexec python -m medos_trainer "$@"\n' > /usr/local/bin/medos-trainer \
 && chmod 0755 /usr/local/bin/medos-trainer

# ---- the build stamp: entry 82's two keys, recorded rather than asserted ------------ #
# `build.sh` reads these out of git. The guard below is why a hand
# `docker build` cannot produce an image that lies about its provenance: it
# fails, loudly, naming the script.
ARG MEDOS_CODE_COMMIT=""
ARG MEDOS_CODE_DIRTY=""
RUN test -n "${MEDOS_CODE_COMMIT}" -a -n "${MEDOS_CODE_DIRTY}" || ( \
      echo "REFUSED: MEDOS_CODE_COMMIT and MEDOS_CODE_DIRTY are build arguments and" >&2; \
      echo "this image will not build without them. The stamp exists so code_commit" >&2; \
      echo "is recorded rather than asserted, and register entry 82 records what a" >&2; \
      echo "default does to it. Build with trainer/build.sh." >&2; \
      exit 1 )
RUN python -m medos_trainer.stamp \
      --code-commit "${MEDOS_CODE_COMMIT}" \
      --code-dirty "${MEDOS_CODE_DIRTY}" \
      --out /opt/medos-trainer/build-stamp.json \
 && cat /opt/medos-trainer/build-stamp.json

# Cross-referenceable from `docker inspect` without running the image. The
# content digest is NOT a label: it is computed during the build from the
# installed set, and a label would have to be a second, hand-kept copy of it.
LABEL org.opencontainers.image.title="medlange-trainer" \
      org.opencontainers.image.revision="${MEDOS_CODE_COMMIT}" \
      dev.medlange.trainer.code-dirty="${MEDOS_CODE_DIRTY}" \
      dev.medlange.trainer.backend="vanilla"

# A separate deployable with a separate identity. Non-root.
RUN useradd --create-home --uid 10001 medlange \
 && mkdir -p /var/lib/medlange-trainer \
 && chown -R medlange:medlange /var/lib/medlange-trainer /opt/medlange-trainer
USER medlange
ENV MEDOS_TRAINER_STAMP=/opt/medlange-trainer/build-stamp.json

# No ENTRYPOINT with a fixed subcommand: the six subcommands are equal
# citizens and the caller names the one it wants.
ENTRYPOINT ["/usr/local/bin/medos-trainer"]
CMD ["--help"]
