'''
Unit test for forecast error functions (point and coverage)
in the metrics module
'''

import pytest
import numpy as np
import pandas as pd

from forecast_tools import metrics as m
from forecast_tools import datasets
from forecast_tools import baseline


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
git
    test calcs produced using libre office calc.
    '''
    error = m.mean_absolute_scaled_error(y_true, y_pred, y_train, period=7)
    assert pytest.approx(expected) == error


@pytest.mark.parametrize("y_intervals, y_test, alpha, expected",
                         [([744.54, 773.22], 741.84, 0.2, 55.68),
                          ([744.54, 773.22], [741.84], 0.2, 55.68),
                          ([744.54, 773.22], 745.0, 0.2, 28.68),
                          (np.array([744.54, 773.22]), 745.0, 0.2, 28.68),
                          (pd.Series([744.54, 773.22]), 745.0, 0.2, 28.68),
                          (pd.DataFrame([744.54, 773.22]).T, 745.0, 0.2, 28.68)])
def test_winkler_score(y_intervals, y_test, alpha, expected):
    '''
    Test that the winkler score returns the correct value

    Tests one step forecasts only.
    '''
    ws = m.winkler_score(y_test, y_intervals, alpha)
    assert pytest.approx(expected) == ws


def test_winkler_score_m_step():
    '''
    test the correct error functions are returned
    '''
    HOLDOUT = 7
    PERIOD = 7
    expected = 130.75

    attends = datasets.load_emergency_dept()

    # train-test split
    train, test = attends[: -HOLDOUT], attends[-HOLDOUT:]

    model = baseline.SNaive(PERIOD)

    # returns 80 and 90% prediction intervals by default.
    preds, intervals = model.fit_predict(train, HOLDOUT,
                                         return_predict_int=True)

    ws = m.winkler_score(test, intervals[0], alpha=0.2)

    assert pytest.approx(expected, abs=0.01) == ws


@pytest.mark.parametrize("y_intervals, y_test, alpha",
                         [([744.54, 773.22], "741.84", 0.2)])
def test_winkler_score_invalid_type(y_intervals, y_test, alpha):
    with pytest.raises(TypeError):
        m.winkler_score(y_test, y_intervals, alpha)


@pytest.mark.parametrize("alpha", [-0.1, 0, 1, 1.1])
def test_winkler_score_invalid_alpha(alpha):
    """Test error thrown if invalid alpha passed to winkler score"""
    with pytest.raises(ValueError):
        m.winkler_score(741.84, [744.54, 773.22], alpha)


def test_winkler_score_invalid_interval():
    """Test that winkler catch incorrect lower > upper bound"""
    with pytest.raises(ValueError):
        m.winkler_score(741.84, [773.22, 744.54], 0.2)


def test_winkler_score_empty_input():
    with pytest.raises(ValueError):
        m.winkler_score([], [], 0.2) == 0.0

def test_winkler_score_mismatched_lengths():
    with pytest.raises(ValueError):
        m.winkler_score([741.84, 760, 770], [[744.54, 773.22], [750, 780]], 0.2)


def test_acd():
    intervals = np.array([[37520, 58225],
                          [29059, 49764],
                          [47325, 68030],
                          [36432, 57137],
                          [35865, 56570],
                          [33419, 54124]])

    y_true = np.array([37463, 40828, 56148, 45342, 43741, 45907])

    acd = m.absolute_coverage_difference(y_true, intervals, alpha=0.05)
    assert pytest.approx(acd, abs=0.01) == 0.12



@pytest.mark.parametrize("pred_intervals, y_true, expected", [
    # Floating point precision tests
    ([[10.0, 15.0], [15.0, 20.0], [25.0, 30.0]], [10.0000001, 20.0, 29.9999999], 1.0),
    
    # Extreme values
    (
        [
            [np.finfo(float).max - 1, np.finfo(float).max],
            [np.finfo(float).min, np.finfo(float).min + 1]
        ],
        [np.finfo(float).max, np.finfo(float).min],
        1.0
    ),
])
def test_coverage_edge_cases(pred_intervals, y_true, expected):
    """Test coverage with floating point precision issues and extreme values."""
    result = m.coverage(y_true, pred_intervals)
    assert pytest.approx(expected) == result

@pytest.mark.parametrize("pred_intervals, y_true", [
    # Invalid types
    ("not an array", [10, 20, 30]),
    ([[5, 15], [15, 25], [25, 35]], "not an array"),
])
def test_coverage_type_errors(pred_intervals, y_true):
    """Test that coverage raises TypeError for invalid input types."""
    with pytest.raises(TypeError):
        m.coverage(y_true, pred_intervals)

@pytest.mark.parametrize("pred_intervals, y_true", [
    # Empty arrays
    ([], []),
    
    # Mismatched lengths
    ([[5, 15], [15, 25]], [10, 20, 30]),
    
    # Invalid intervals (lower > upper)
    ([[15, 5], [25, 15], [35, 25]], [10, 20, 30]),
    
    # Wrong interval shape
    ([[5, 15, 20], [15, 25, 30], [25, 35, 40]], [10, 20, 30]),
])
def test_coverage_value_errors(pred_intervals, y_true):
    """Test that coverage raises ValueError for invalid inputs."""
    with pytest.raises(ValueError):
        m.coverage(y_true, pred_intervals)