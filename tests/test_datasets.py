import pytest
from forecast_tools import datasets as ds


@pytest.mark.parametrize("y_true, y_pred, metrics, expected", 
def test_load_ed_shape(expected):
    df = ds.load_emergency_dept()
