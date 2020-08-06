'''
Unit test for forecast error functions (point and coverage)
in the metrics module
'''

import pytest
import numpy as np

from forecast_tools import metrics as m


@pytest.mark.parametrize("y_true, y_pred, metrics, expected",
                         [([1], [1], 'all', 6),
                          ([1], [1], ['mae'], 1),
                          ([1], [1], ['mae', 'me'], 2),
                          ([1], [1], ['mae', 'me', 'smape'], 3),
                          ([1], [1], ['mae', 'me', 'smape', 'mse',
                                      'rmse', 'mape'], 6)])
def test_forecast_error_return_length(y_true, y_pred, metrics, expected):
    '''
    test the correct number of error metric functions are returned.
    '''
    funcs_dict = m.forecast_errors(y_true, y_pred, metrics)
    assert len(funcs_dict) == expected


@pytest.mark.parametrize("y_true, y_pred, metrics, expected",
                         [([1], [1], 'all', ['me', 'mae', 'mse', 'rmse',
                                             'mape', 'smape']),
                          ([1], [1], ['mae'], ['mae']),
                          ([1], [1], ['mae', 'me'], ['mae', 'me']),
                          ([1], [1], ['mae', 'me', 'smape'], ['mae', 'me',
                                                              'smape'])])
def test_forecast_error_return_funcs(y_true, y_pred, metrics, expected):
    '''
    test the correct error functions are returned
    '''
    funcs_dict = m.forecast_errors(y_true, y_pred, metrics)
    assert list(funcs_dict.keys()) == expected


@pytest.mark.parametrize("y_pred, y_true, expected",
                         [([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], 0.0),
                          ([1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], 6.0),
                          ([103, 130, 132, 124, 124, 108],
                           [129, 111, 122, 129, 110, 141], 17.833333),
                          ([103, 130, 132, 124, 124, 108, 160, 160],
                           [129, 111, 122, 129, 110, 141, 142, 143], 17.75)])
def test_mean_absolute_error(y_true, y_pred, expected):
    '''
    test mean absolute error calculation
    '''
    error = m.mean_absolute_error(y_true, y_pred)
    assert pytest.approx(expected) == error


@pytest.mark.parametrize("y_pred, y_true, expected",
                         [([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], 0.0),
                          ([1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], 6.0),
                          ([103, 130, 132, 124, 124, 108],
                           [129, 111, 122, 129, 110, 141], 3.5),
                          ([103, 130, 132, 124, 124, 108, 160, 160],
                           [129, 111, 122, 129, 110, 141, 142, 143], -1.75)])
def test_mean_error(y_true, y_pred, expected):
    '''
    test mean error calculation
    '''
    error = m.mean_error(y_true, y_pred)
    assert pytest.approx(expected) == error


@pytest.mark.parametrize("y_pred, y_true, expected",
                         [([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], 0.0),
                          ([1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12],
                           65.3210678210678),
                          ([103, 130, 132, 124, 124, 108],
                           [129, 111, 122, 129, 110, 141],
                           14.2460623711587),
                          ([103, 130, 132, 124, 124, 108, 160, 160],
                           [129, 111, 122, 129, 110, 141, 142, 143],
                           13.7550678066365)])
def test_mean_absolute_percentage_error(y_true, y_pred, expected):
    '''
    test mean error calculation
    '''
    error = m.mean_absolute_percentage_error(y_true, y_pred)
    assert pytest.approx(expected) == error


@pytest.mark.parametrize("y_pred, y_true, expected",
                         [([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], 0.0),
                          ([1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12],
                           36.0),
                          ([103, 130, 132, 124, 124, 108],
                           [129, 111, 122, 129, 110, 141],
                           407.833333333333),
                          ([103, 130, 132, 124, 124, 108, 160, 160],
                           [129, 111, 122, 129, 110, 141, 142, 143],
                           382.50)])
def test_mean_squared_error(y_true, y_pred, expected):
    '''
    test mean squared error calculation
    '''
    error = m.mean_squared_error(y_true, y_pred)
    assert pytest.approx(expected) == error


@pytest.mark.parametrize("y_pred, y_true, expected",
                         [([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], 0.0),
                          ([1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12],
                           6.0),
                          ([103, 130, 132, 124, 124, 108],
                           [129, 111, 122, 129, 110, 141],
                           20.1948838405506),
                          ([103, 130, 132, 124, 124, 108, 160, 160],
                           [129, 111, 122, 129, 110, 141, 142, 143],
                           19.5576072156079)])
def test_root_mean_squared_error(y_true, y_pred, expected):
    '''
    test root mean squared error calculation
    '''
    error = m.root_mean_squared_error(y_true, y_pred)
    assert pytest.approx(expected) == error


@pytest.mark.parametrize("y_pred, y_true, expected",
                         [([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], 0.0),
                          ([1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12],
                           99.5634920634921),
                          ([103, 130, 132, 124, 124, 108],
                           [129, 111, 122, 129, 110, 141],
                           14.7466414897349),
                          ([103, 130, 132, 124, 124, 108, 160, 160],
                           [129, 111, 122, 129, 110, 141, 142, 143],
                           13.9526876064932)])
def test_symmetric_mape(y_true, y_pred, expected):
    '''
    test symmetric mean absolute percentage error calculation
    '''
    error = m.symmetric_mean_absolute_percentage_error(y_true, y_pred)
    assert pytest.approx(expected) == error


@pytest.mark.parametrize("y_true, y_intervals, expected",
                         [([10, 20, 30, 40, 50],
                           [[5, 15, 25, 35, 45],
                            [15, 25, 35, 45, 55]],
                           1.0),
                          ([20, 20, 30, 40, 50],
                           [[5, 15, 25, 35, 45],
                            [15, 25, 35, 45, 55]],
                           0.8),
                          ([20, 30, 30, 40, 50],
                           [[5, 15, 25, 35, 45],
                            [15, 25, 35, 45, 55]],
                           0.6),
                          ([20, 20, 30, 40, 30],
                           [[5, 15, 25, 35, 45],
                            [15, 25, 35, 45, 55]],
                           0.6),
                          ([100, 100, 100, 100, 100],
                           [[5, 15, 25, 35, 45],
                            [15, 25, 35, 45, 55]],
                           0.0)])
def test_coverage(y_true, y_intervals, expected):
    '''
    test prediction interval coverage
    '''
    y_intervals = np.array(y_intervals).T
    error = m.coverage(y_true, y_intervals)
    assert pytest.approx(expected) == error


@pytest.mark.parametrize("y_train, y_pred, y_true, expected",
                         [(np.arange(10), [1, 2, 3, 4, 5, 6],
                          [1, 2, 3, 4, 5, 6], 0.0),
                          (np.arange(1, 21), np.arange(21, 26),
                           np.full(5, 10), 13)])
def test_mase_naive(y_train, y_true, y_pred, expected):
    '''
    test mean absolute scaled error calculation using naive as scaler.

    test calcs produced using libre office calc.
    '''
    error = m.mean_absolute_scaled_error(y_true, y_pred, y_train, period=None)
    assert pytest.approx(expected) == error


@pytest.mark.parametrize("y_train, y_pred, y_true, expected",
                         [(np.arange(1, 21), [1, 2, 3, 4, 5, 6],
                          [1, 2, 3, 4, 5, 6], 0.0),

                          (np.arange(1, 21), np.arange(21, 26),
                           np.full(5, 10), 1.85714286)])
def test_mase_snaive(y_train, y_true, y_pred, expected):
    '''
    test mean absolute scaled error calculation using SNaive as scaler

    test calcs produced using libre office calc.
    '''
    error = m.mean_absolute_scaled_error(y_true, y_pred, y_train, period=7)
    assert pytest.approx(expected) == error
