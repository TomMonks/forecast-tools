import pytest

from basecast.metrics import (mean_absolute_error,
                              mean_absolute_percentage_error, 
                              mean_error, 
                              mean_squared_error, 
                              root_mean_squared_error, 
                              symmetric_mean_absolute_percentage_error, 
                              forecast_errors)

@pytest.mark.parametrize("y_true, y_pred, metrics, expected", 
                         [([1], [1], 'all', 6),
                          ([1], [1], ['mae'], 1),
                          ([1], [1], ['mae', 'me'], 2),
                          ([1], [1], ['mae', 'me', 'smape'], 3),
                          ([1], [1], ['mae', 'me', 'smape', 'mse', 'rmse', 'mape'], 6)])
def test_forecast_error_return_length(y_true, y_pred, metrics, expected):
    funcs_dict = forecast_errors(y_true, y_pred, metrics)
    assert len(funcs_dict) == expected