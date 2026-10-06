# SPDX-License-Identifier: Apache-2.0
"""DICOM series import: one series in, one `.npz` case out.

THE SYNTHETIC SERIES are built with pydicom from scratch — minimal tags,
8x8 pixels, axial orientation — and written with filenames that lie about
slice order, because the whole point of the importer's sorting is to not
believe them.
"""

from __future__ import annotations

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

from medos_trainer.__main__ import main  # noqa: E402
from medos_trainer.standalone import import_dicom_series  # noqa: E402
from pydicom.dataset import Dataset, FileMetaDataset  # noqa: E402
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid  # noqa: E402


def _write_slice(path, series_uid, k, slope="2", intercept="-100"):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.ImplementationClassUID = generate_uid()
    ds = Dataset()
    ds.file_meta = meta
    ds.Modality = "CT"
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = generate_uid()
    ds.ImagePositionPatient = [0.0, 0.0, float(k) * 2.5]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.PixelSpacing = [0.7, 0.5]
    ds.RescaleSlope = slope
    ds.RescaleIntercept = intercept
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = 8
    ds.Columns = 8
    ds.PixelData = np.full((8, 8), k, dtype=np.int16).tobytes()
    ds.save_as(str(path), enforce_file_format=True)


def _write_series(directory, n=5, series_uid=None, slope="2", intercept="-100",
                  shuffled_names=True):
    directory.mkdir(exist_ok=True)
    series_uid = series_uid or generate_uid()
    # Filenames deliberately do not encode position order.
    names = ["slice-c", "slice-a", "slice-e", "slice-b", "slice-d"][:n]
    for k in range(n):
        file_name = names[k] if shuffled_names else f"slice-{k:02d}"
        _write_slice(directory / f"{file_name}.dcm", series_uid, k,
                     slope=slope, intercept=intercept)
    return series_uid


def test_import_round_trips_shape_spacing_rescale_and_order(tmp_path) -> None:
    series = tmp_path / "series"
    _write_series(series)
    out = tmp_path / "case.npz"
    n = import_dicom_series(series, out)
    assert n == 1
    with np.load(out) as z:
        image = z["image"]
        spacing = z["spacing_mm"]
    assert image.shape == (1, 5, 8, 8)
    assert image.dtype == np.float32
    # (K slice spacing from sorted positions, J PixelSpacing[0], I PixelSpacing[1])
    assert list(spacing) == [2.5, 0.7, 0.5]
    # Slice k holds the constant k in every pixel: ordering, rescale applied.
    plane = (np.arange(5, dtype=np.float32) * 2.0 - 100.0)[:, None, None] * np.ones(
        (1, 8, 8), np.float32
    )
    assert np.array_equal(image[0], plane)


def test_import_defaults_rescale_to_identity(tmp_path) -> None:
    series = tmp_path / "series"
    _write_series(series, slope="1", intercept="0")
    out = tmp_path / "case.npz"
    import_dicom_series(series, out)
    with np.load(out) as z:
        assert np.array_equal(z["image"][0, :, 0, 0], np.arange(5, dtype=np.float32))


def test_two_series_are_refused_by_uid(tmp_path) -> None:
    series = tmp_path / "series"
    series.mkdir()
    first = generate_uid()
    second = generate_uid()
    _write_slice(series / "a-1.dcm", first, 0)
    _write_slice(series / "a-2.dcm", first, 1)
    _write_slice(series / "b-1.dcm", second, 0)
    with pytest.raises(ValueError) as excinfo:
        import_dicom_series(series, tmp_path / "case.npz")
    message = str(excinfo.value)
    assert first in message and second in message


def test_unreadable_directory_is_named(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "notes.txt").write_text("not dicom at all")
    with pytest.raises(ValueError, match="no readable DICOM"):
        import_dicom_series(empty, tmp_path / "case.npz")


def test_import_dicom_cli_smoke(tmp_path) -> None:
    series = tmp_path / "series"
    _write_series(series)
    out = tmp_path / "case.npz"
    rc = main(["vanilla-import-dicom", "--series", str(series), "--out", str(out)])
    assert rc == 0
    with np.load(out) as z:
        assert z["image"].shape == (1, 5, 8, 8)
