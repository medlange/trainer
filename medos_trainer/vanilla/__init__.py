# SPDX-License-Identifier: Apache-2.0
"""Medlange Trainer — the vanilla stack.

Pure-PyTorch replacement for the nnU-Net/MONAI inheritance: networks, data,
training, inference and planning written from scratch (roadmap phase T-vanilla,
owner decision 2026-10-04, grounds in docs/audits/trainer-2026-10-04.md).

THE REASON THIS PACKAGE EXISTS. The audit's verdict was that the trainer is a
"disciplined derivative" — its value sat on top of two frameworks it did not
control. Vanilla changes the question from "how do we wrap nnU-Net" to "what
does Medlange Trainer itself do": every behaviour here is ours, reviewable in
this repository, and fixable without a dependency release cycle.
"""
