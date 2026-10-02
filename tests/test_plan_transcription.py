# SPDX-License-Identifier: Apache-2.0
"""`MOS-TRAIN-223`'s transcription, against REAL plan documents from both architectures.

WHY REAL DOCUMENTS AND NOT A CONSTRUCTED ONE
---------------------------------------------
`export_spec_fields` REFUSES any derived quantity it cannot map exactly, which is the right
behaviour and makes it the single point where a new architecture breaks the pipeline. It
breaks there LATE: the transcription runs at the end of the plan phase, after the fingerprint
extraction and the preprocessing, and on this cohort a plan phase is twenty minutes.

A hand-built fixture cannot establish that a real ResEnc plan transcribes, because a fixture
is written by someone who already knows which keys the exporter reads. The two documents here
were produced by nnU-Net itself -- `ExperimentPlanner` and `nnUNetPlannerResEncL`, both
against this cohort's own fingerprint -- and are stored verbatim. Trimming them to the keys
the exporter happens to read today would make this gate pass while the real document failed,
which is the exact shape of defect the rest of this suite spent a night finding.

They carry cohort-level aggregate statistics only: spacings, median shapes, and foreground
intensity percentiles over 585 cases. No case identifier, no date, no institution, no path.

WHAT THE SHARED `data_identifier` MEANS, AND WHY IT IS PINNED HERE
------------------------------------------------------------------
Both documents name `nnUNetPlans_3d_fullres` as the `data_identifier` of their `3d_fullres`
configuration, because their PREPROCESSING parameters are identical -- same spacing, same
normalisation, same mask-for-norm -- so nnU-Net reuses one preprocessed folder for both.

Two consequences, and both matter:

  * comparing the architectures costs two fits and no preprocessing;
  * the two arms read byte-identical preprocessed data, so `ab_evaluate.comparable()` -- which
    refuses to subtract two measurements whose `inputs_digest` differs -- will accept them.

If a future nnU-Net stops sharing that folder, both consequences evaporate silently: the
comparison would still run, and it would be measuring two different samples. Hence the
assertion.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TRAINER = Path(__file__).resolve().parents[1]
if str(TRAINER) not in sys.path:
    sys.path.insert(0, str(TRAINER))

FIXTURES = Path(__file__).resolve().parent / "fixtures"

from medos_trainer import architectures as cat  # noqa: E402

#: The preset each fixture was produced by, so the catalogue's claims are checked against a
#: document that planner actually wrote rather than against the catalogue itself.
DOCUMENTS = {
    "plain": FIXTURES / "nnunet_plans_plain.json",
    "resenc_l": FIXTURES / "nnunet_plans_resenc_l.json",
}


def _plans(preset_name: str) -> dict:
    return json.loads(DOCUMENTS[preset_name].read_text(encoding="utf-8"))


@pytest.mark.parametrize("preset_name", sorted(DOCUMENTS))
def test_the_transcription_accepts_the_document_the_planner_wrote(preset_name) -> None:
    """The gate that would otherwise fail twenty minutes into a plan phase."""
    from medos.sdk.autoconfig import export_spec_fields
    from medos_trainer.backend import single_channel_view

    document = _plans(preset_name)
    exported = export_spec_fields(single_channel_view(document, "3d_fullres"), backend="nnunet")
    assert sorted(exported) == [
        "backend", "fingerprint_digest", "hyperparameters", "spec_fields",
    ]
    assert exported["backend"] == "nnunet"
    assert len(exported["spec_fields"]) == 10, exported["spec_fields"]
    # MOS-TRAIN-224: the TRAINING batch size, which must never land in the spec's patch block.
    assert exported["hyperparameters"] == {"training_batch_size": 2}
    assert "batch_size" not in exported["spec_fields"].get("patch", {})


@pytest.mark.parametrize("preset_name", sorted(DOCUMENTS))
def test_the_two_fingerprint_digests_are_two_quantities_and_differ(preset_name) -> None:
    """TWO DIGESTS, ONE ENGLISH NAME -- the shape of hazard `MOS-TRAIN-224` names for the two
    batch sizes, here for the fingerprint.

    `fingerprint_digest(plans)` digests the VERBATIM document, which `MOS-TRAIN-223` retains
    for audit and which `derive_plan` writes into `plan.json` as `fingerprint_digest`.
    `export_spec_fields(...)["fingerprint_digest"]` digests the SINGLE-CHANNEL VIEW the
    exporter reads. They are different bytes and therefore different digests, and the one the
    platform freezes is the first.

    Asserted so that a future edit which started recording the exporter's digest under the
    frozen field would be caught here rather than by a reviewer comparing a run's recorded
    digest against a document and finding they disagree with no explanation.
    """
    from medos.sdk.autoconfig import export_spec_fields, fingerprint_digest
    from medos_trainer.backend import single_channel_view

    document = _plans(preset_name)
    verbatim = fingerprint_digest(document)
    view = export_spec_fields(single_channel_view(document, "3d_fullres"), backend="nnunet")[
        "fingerprint_digest"
    ]
    assert verbatim != view, (
        "the verbatim document and the single-channel view digest alike, so either the view "
        "no longer drops a channel or one of the two digests is not over what it claims"
    )
    assert verbatim.startswith("sha256:") and view.startswith("sha256:")


@pytest.mark.parametrize("preset_name", sorted(DOCUMENTS))
def test_the_document_names_the_planner_and_the_network_the_catalogue_claims(
    preset_name
) -> None:
    """The catalogue's `plans_identifier` and `network` are what `plan.json` records. Checked
    against the document the planner wrote, so the record cannot name an architecture the run
    did not train."""
    preset = cat.named(preset_name)
    document = _plans(preset_name)
    assert document["plans_name"] == preset.plans_identifier
    network = document["configurations"]["3d_fullres"]["architecture"]["network_class_name"]
    assert network.rsplit(".", 1)[-1] == preset.network, network


def test_both_architectures_share_one_preprocessed_folder() -> None:
    """THE SUBTLE PIN. See the module docstring: this is what makes the architecture
    comparison cheap AND comparable, and if it stops holding both facts fail quietly."""
    identifiers = {
        name: _plans(name)["configurations"]["3d_fullres"]["data_identifier"]
        for name in DOCUMENTS
    }
    assert len(set(identifiers.values())) == 1, (
        f"the two architectures name different preprocessed folders: {identifiers}. "
        "Comparing them now needs its own preprocessing pass, and the two arms would read "
        "different bytes -- which ab_evaluate.comparable() exists to refuse"
    )

    preprocessing = {}
    for name in DOCUMENTS:
        configuration = _plans(name)["configurations"]["3d_fullres"]
        preprocessing[name] = {
            key: configuration[key] for key in
            ("spacing", "normalization_schemes", "use_mask_for_norm", "preprocessor_name")
        }
    values = list(preprocessing.values())
    assert values[0] == values[1], (
        f"the folder is shared but the preprocessing parameters differ: {preprocessing}. "
        "One of the two is then reading data derived under the other's rules"
    )


def test_the_two_documents_are_distinguished_by_their_digest() -> None:
    """`fingerprint_digest` is what `plan.json` records as `fingerprint_digest` and what the
    platform freezes. Two architectures sharing a digest would make the frozen record unable
    to say which one was planned."""
    from medos.sdk.autoconfig import fingerprint_digest

    digests = {name: fingerprint_digest(_plans(name)) for name in DOCUMENTS}
    assert len(set(digests.values())) == len(digests), digests


def test_the_architectures_differ_where_the_preset_says_they_do() -> None:
    """What the preset actually buys, asserted rather than assumed: a deeper residual encoder,
    a lighter decoder, and a taller patch at the same batch size.

    Asserted because "we switched architecture" is a claim about the plan, and a preset that
    silently produced the same network as the baseline would make the whole comparison a
    measurement of noise.
    """
    plain = _plans("plain")["configurations"]["3d_fullres"]
    resenc = _plans("resenc_l")["configurations"]["3d_fullres"]

    assert plain["batch_size"] == resenc["batch_size"], (
        "the batch sizes differ, so a difference in results could be the batch rather than "
        "the architecture"
    )
    assert resenc["patch_size"][0] > plain["patch_size"][0], (
        f"the residual encoder was expected to afford a taller patch: {plain['patch_size']} "
        f"vs {resenc['patch_size']}"
    )
    plain_kwargs = plain["architecture"]["arch_kwargs"]
    resenc_kwargs = resenc["architecture"]["arch_kwargs"]
    assert plain_kwargs.get("n_blocks_per_stage") is None
    assert resenc_kwargs.get("n_conv_per_stage") is None
    assert sum(resenc_kwargs["n_blocks_per_stage"]) > sum(plain_kwargs["n_conv_per_stage"]), (
        "the residual encoder is not deeper than the plain one, so the preset is buying "
        "nothing this comparison could attribute to it"
    )
    assert resenc_kwargs["features_per_stage"] == plain_kwargs["features_per_stage"], (
        "the feature widths differ as well; a difference in results would then have two "
        "causes and the comparison could not separate them"
    )


def test_the_fixtures_carry_no_case_identifier_or_date() -> None:
    """The documents are stored verbatim, so the claim that they hold only cohort-level
    aggregates is worth asserting rather than asserting once in a docstring and trusting.

    A future nnU-Net that started writing case names or source paths into its plans would
    make these fixtures a disclosure, and this is the cheapest place to find that out.
    """
    forbidden = (
        "patientname", "patientid", "patientbirthdate", "accessionnumber", "studydate",
        "studyinstanceuid", "seriesinstanceuid", "institution", "stationname", "mrn",
        "/data/", "/work/", "c:\\", "zu-dev",
    )
    for name, path in DOCUMENTS.items():
        text = path.read_text(encoding="utf-8").lower()
        present = [needle for needle in forbidden if needle in text]
        assert not present, f"{name} carries {present}"
        document = json.loads(path.read_text(encoding="utf-8"))
        # Aggregate statistics per CHANNEL, never per case: one entry for the single CT input.
        assert sorted(document["foreground_intensity_properties_per_channel"]) == ["0"]
