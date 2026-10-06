# SPDX-License-Identifier: Apache-2.0
"""Medlange Trainer — a vanilla-PyTorch framework for volumetric segmentation.

WHAT THIS PACKAGE IS
---------------------
An nnU-Net-class trainer written from scratch: its own 3D UNet with deep
supervision (`vanilla.nets`), its own fingerprint-based planning
(`vanilla.plan`), its own masked-loss training loop (`vanilla.trainer` +
`vanilla.losses`), its own sliding-window inference (`vanilla.infer`), its
own serving export (`vanilla.export`) and its own case format
(`vanilla.data`). No nnU-Net, no MONAI, no platform: the package imports
nothing outside itself but torch, numpy, scipy and — lazily, for the
importers and the ONNX branch — nibabel, pydicom and onnx.

THE MODULES

    vanilla/     the framework: data, plan, nets, losses, trainer, infer, export
    standalone   the autonomous entry: cases in, plan/fit/crossval/evaluate out,
                 plus the DICOM and NIfTI importers
    stamp        the build stamp, recorded at `docker build` and read back
                 here. Stdlib only, because it runs inside the build
    environment  `MEDOS_TRAINING_ENVIRONMENT`'s nine keys, OBSERVED rather
                 than declared wherever observing them is possible
    detection    blob-detection metric utilities (scipy/numpy, pure)
    evidence     evidence-document helpers (pure)
    overlap      overlap metrics (pure)
    __main__     the CLI: vanilla-plan / vanilla-fit / vanilla-crossval /
                 vanilla-evaluate / vanilla-import-nnunet /
                 vanilla-import-dicom / export / predict /
                 declare-environment / doctor
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
