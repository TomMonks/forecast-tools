import pytest

from forecast_tools import datasets as ds


def test_load_ed_shape():
    expected_shape = (344, 1)
    df = ds.load_emergency_dept()
    assert df.shape == expected_shape


def test_load_ed_freq():
    expected_freq = 'D'
    df = ds.load_emergency_dept()
    assert df.index.freq.freqstr == expected_freq
