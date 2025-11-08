import os
import glob
import numpy as np
import pytest

from ligotools import readligo as rl

DATA_DIR = "data"


def _get_any_hdf5():
    """Return the first .hdf5 file in data/ for testing, or None."""
    files = glob.glob(os.path.join(DATA_DIR, "*.hdf5"))
    return files[0] if files else None


def test_read_hdf5_loads_data():
    fname = _get_any_hdf5()
    if fname is None:
        pytest.skip("No .hdf5 files in data/ — skipping test.")

    out = rl.read_hdf5(fname)

    # we know it's a tuple with at least strain + some metadata
    assert isinstance(out, tuple)
    assert len(out) >= 2

    strain = out[0]
    gps_start = out[1]

    # 1) strain should be an array
    assert isinstance(strain, np.ndarray)
    assert strain.size > 0

    # 2) second value in your version is a single GPS start time (np.int64)
    assert isinstance(gps_start, (int, np.integer))


def test_loaddata_returns_expected_shapes():
    fname = _get_any_hdf5()
    if fname is None:
        pytest.skip("No .hdf5 files in data/ — skipping test.")

    detector = "H1" if os.path.basename(fname).startswith("H-") else "L1"

    out = rl.loaddata(fname, detector)

    assert isinstance(out, tuple)
    assert len(out) >= 3  # strain, time, metadata/dict

    strain = out[0]
    time = out[1]
    meta = out[2]

    assert isinstance(strain, np.ndarray)
    assert isinstance(time, np.ndarray)
    assert len(strain) == len(time)
    assert isinstance(meta, dict)
