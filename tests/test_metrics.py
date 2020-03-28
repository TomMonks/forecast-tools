import pytest

from forecast_tools import metrics as m
                              
@pytest.mark.parametrize("y_true, y_pred, metrics, expected", 
                         [([1], [1], 'all', 6),
                          ([1], [1], ['mae'], 1),
                          ([1], [1], ['mae', 'me'], 2),
                          ([1], [1], ['mae', 'me', 'smape'], 3),
                          ([1], [1], ['mae', 'me', 'smape', 'mse', 
                                      'rmse', 'mape'], 6)])
def test_forecast_error_return_length(y_true, y_pred, metrics, expected):
    funcs_dict = m.forecast_errors(y_true, y_pred, metrics)
    assert len(funcs_dict) == expected


@pytest.mark.parametrize("y_true, y_pred, metrics, expected", 
                         [([1], [1], 'all', ['me', 'mae', 'mse', 'rmse', 
                                             'mape','smape']),
                          ([1], [1], ['mae'], ['mae']),
                          ([1], [1], ['mae', 'me'], ['mae', 'me']),
                          ([1], [1], ['mae', 'me', 'smape'], ['mae', 'me', 
                                                              'smape'])])
def test_forecast_error_return_funcs(y_true, y_pred, metrics, expected):
    funcs_dict = m.forecast_errors(y_true, y_pred, metrics)
    assert list(funcs_dict.keys()) == expected