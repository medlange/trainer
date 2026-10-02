# SPDX-License-Identifier: Apache-2.0
"""`write_modelcard`: the trainer's half of the card contract.

The reader half lives in the platform suite (`tests/unit/test_modelcard.py`), because the
round trip crosses the trainer/platform boundary and the platform's suite may read both
trees. This file asserts what the trainer owns: the card it writes at the end of `fit`
carries the derived spec, the weights facts the bundle report verified, the framework
versions that produced them, and an outputs descriptor derived from the REGISTERED spec's
label set -- not from nnU-Net's own dataset.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from medos.sdk.fixtures import selftest_spec_document
from medos.sdk.modelcard import CARD_FILENAME, CARD_FORMAT
from medos_trainer import packaging


def test_write_modelcard_emits_a_card_the_sdk_format_describes(tmp_path: Path) -> None:
    spec_document = selftest_spec_document()
    packaging.write_modelcard(
        tmp_path,
        spec_document=spec_document,
        bundle_dir="bundle",
        weights_file="models/model.ts",
        weights_digest="sha256:" + "b" * 64,
        versions={"torch": "2.7.1", "numpy": "1.26.4"},
        stamp={"code_commit": "c" * 40, "code_dirty": False},
    )

    document = json.loads((tmp_path / CARD_FILENAME).read_text(encoding="utf-8"))

    assert document["format"] == CARD_FORMAT
    assert document["model_id"] == spec_document["model_id"]
    assert document["model_version"] == spec_document["model_version"]
    assert document["preprocessing"]["document"] == spec_document
    assert document["weights"] == {
        "path": "bundle",
        "weights_file": "models/model.ts",
        "format": "torchscript",
        "digest": "sha256:" + "b" * 64,
    }
    assert document["frameworks"]["monai_bundle"] == packaging.MONAI_BUNDLE_TARGET
    assert document["stamp"] == {"code_commit": "c" * 40, "code_dirty": False}


def test_outputs_are_derived_from_the_registered_label_set(tmp_path: Path) -> None:
    spec_document = selftest_spec_document()
    label_set = spec_document["io"]["label_set"]

    packaging.write_modelcard(
        tmp_path,
        spec_document=spec_document,
        bundle_dir="bundle",
        weights_file="models/model.ts",
        weights_digest="sha256:" + "b" * 64,
        versions={"torch": "2.7.1", "numpy": "1.26.4"},
        stamp={"code_commit": "c" * 40},
    )

    document = json.loads((tmp_path / CARD_FILENAME).read_text(encoding="utf-8"))

    assert document["outputs"] == [
        {"kind": "segmentation", "value": entry["value"], "name": entry["name"]}
        for entry in label_set
    ]


def test_the_written_card_loads_through_the_sdk_reader(tmp_path: Path) -> None:
    """The seam both halves meet on, exercised from the trainer's side."""
    from medos.sdk.modelcard import ModelCard

    (tmp_path / "bundle").mkdir()
    spec_document = selftest_spec_document()
    packaging.write_modelcard(
        tmp_path,
        spec_document=spec_document,
        bundle_dir="bundle",
        weights_file="models/model.ts",
        weights_digest="sha256:" + "b" * 64,
        versions={"torch": "2.7.1", "numpy": "1.26.4"},
        stamp={"code_commit": "c" * 40},
    )

    card = ModelCard.load(tmp_path)

    assert card.spec.digest == card.document()["preprocessing"]["digest"]
    assert card.chain().names[0] == "Orientationd"
