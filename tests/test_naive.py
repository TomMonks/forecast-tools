'''
Tests for Naive benchmark classes

Test types:

1. Forecast horizons
2. Allowable input types: np.ndarray, pd.DataFrame, pd.Series
3. Failure paths for abnormal input
4. Predictions
    - naive1 - carries forward last value
    - snaive - carries forward previous h values (to do)
    - average - flat forecast of average
    - drift - previous value + gradient (to do)

5. Prediction intervals
    - horizon 
    - sets i.e. 2 sets of intervals (0.8 and 0.95)
    - width (to do - benchmark against R?)
    - bootstrapped prediction intervals (to do - need to think carefully about test)

6. Fitted values (to do)


'''

import pytest
import pandas as pd
import numpy as np

import forecast_tools.baseline as b

@pytest.mark.parametrize("data, horizon, expected", 
                         [([1, 2, 3, 4, 5], 12, 12),
                          ([1, 2, 3, 4, 5], 24, 24),
                          ([1, 2, 3], 8, 8)
                          ])
def test_naive1_forecast_horizon(data, horizon, expected):
    '''
    test the correct number of error metric functions are returned.
    '''
    model = b.Naive1()
    model.fit(pd.Series(data))
    #point forecasts only
    preds = model.predict(horizon)
    assert len(preds) == expected

@pytest.mark.parametrize("data, horizon, expected", 
                         [([1, 2, 3, 4, 5], 12, 12),
                          ([1, 2, 3, 4, 5], 24, 24),
                          ([1, 2, 3], 8, 8)
                          ])
def test_snaive_forecast_horizon(data, horizon, expected):
    '''
    test the correct number of error metric functions are returned.
    '''
    model = b.SNaive(1)
    model.fit(pd.Series(data))
    #point forecasts only
    preds = model.predict(horizon)
    assert len(preds) == expected


@pytest.mark.parametrize("data, horizon, expected", 
                         [([1, 2, 3, 4, 5], 12, 12),
                          ([1, 2, 3, 4, 5], 24, 24),
                          ([1, 2, 3], 8, 8)
                          ])
def test_drift_forecast_horizon(data, horizon, expected):
    '''
    test the correct number of error metric functions are returned.
    '''
    model = b.Drift()
    model.fit(np.array(data))
    #point forecasts only
    preds = model.predict(horizon)
    assert len(preds) == expected


@pytest.mark.parametrize("data, horizon, expected", 
                         [([1, 2, 3, 4, 5], 12, 12),
                          ([1, 2, 3, 4, 5], 24, 24),
                          ([1, 2, 3], 8, 8)
                          ])
def test_average_forecast_horizon(data, horizon, expected):
    '''
    test the correct number of error metric functions are returned.
    '''
    model = b.Average()
    model.fit(pd.Series(data))
    #point forecasts only
    preds = model.predict(horizon)
    assert len(preds) == expected


@pytest.mark.parametrize("data, horizon, expected", 
                         [([1, 2, 3, 4, 5], 12, 12),
                          ([1, 2, 3, 4, 5], 24, 24),
                          ([1, 2, 3], 8, 8)
                          ])
def test_average_forecast_input_numpy(data, horizon, expected):
    '''
    test the average class accepts numpy array
    '''
    model = b.Average()
    model.fit(np.array(data))
    #point forecasts only
    preds = model.predict(horizon)
    assert len(preds) == expected

@pytest.mark.parametrize("data, horizon, expected", 
                         [([1, 2, 3, 4, 5], 12, 12),
                          ([1, 2, 3, 4, 5], 24, 24),
                          ([1, 2, 3], 8, 8)
                          ])
def test_average_forecast_input_series(data, horizon, expected):
    '''
    test the average class accepts pandas series
    '''
    model = b.Average()
    model.fit(pd.Series(data))
    #point forecasts only
    preds = model.predict(horizon)
    assert len(preds) == expected

@pytest.mark.parametrize("data, horizon, expected", 
                         [([1, 2, 3, 4, 5], 12, 12),
                          ([1, 2, 3, 4, 5], 24, 24),
                          ([1, 2, 3], 8, 8)
                          ])
def test_average_forecast_input_dataframe(data, horizon, expected):
    '''
    test the average baseline class accept dataframe
    '''
    model = b.Average()
    model.fit(pd.DataFrame(data))
    #point forecasts only
    preds = model.predict(horizon)
    assert len(preds) == expected


@pytest.mark.parametrize("data, horizon, expected", 
                         [([1, 2, 3, 4, 5], 12, 12),
                          ([1, 2, 3, 4, 5], 24, 24),
                          ([1, 2, 3], 8, 8)
                          ])
def test_naive1_forecast_input_dataframe(data, horizon, expected):
    '''
    test the correct number of error metric functions are returned.
    '''
    model = b.Naive1()
    model.fit(pd.DataFrame(data))
    #point forecasts only
    preds = model.predict(horizon)
    assert len(preds) == expected


@pytest.mark.parametrize("data, horizon, expected", 
                         [([1, 2, 3, 4, 5], 12, 12),
                          ([1, 2, 3, 4, 5], 24, 24),
                          ([1, 2, 3], 8, 8)
                          ])
def test_naive1_forecast_input_series(data, horizon, expected):
    '''
    test the correct number of error metric functions are returned.
    '''
    model = b.Naive1()
    model.fit(pd.Series(data))
    #point forecasts only
    preds = model.predict(horizon)
    assert len(preds) == expected

@pytest.mark.parametrize("data, horizon, expected", 
                         [([1, 2, 3, 4, 5], 12, 12),
                          ([1, 2, 3, 4, 5], 24, 24),
                          ([1, 2, 3], 8, 8)
                          ])
def test_naive1_forecast_input_numpy(data, horizon, expected):
    '''
    test the correct number of error metric functions are returned.
    '''
    model = b.Naive1()
    model.fit(np.array(data))
    #point forecasts only
    preds = model.predict(horizon)
    assert len(preds) == expected


@pytest.mark.parametrize("data, horizon, expected", 
                         [([1, 2, 3, 4, 5], 12, 12),
                          ([1, 2, 3, 4, 5], 24, 24),
                          ([1, 2, 3], 8, 8)
                          ])
def test_snaive_forecast_input_series(data, horizon, expected):
    '''
    test the correct number of error metric functions are returned.
    '''
    model = b.SNaive(1)
    model.fit(pd.Series(data))
    #point forecasts only
    preds = model.predict(horizon)
    assert len(preds) == expected


@pytest.mark.parametrize("data, horizon, expected", 
                         [([1, 2, 3, 4, 5], 12, 12),
                          ([1, 2, 3, 4, 5], 24, 24),
                          ([1, 2, 3], 8, 8)
                          ])
def test_snaive_forecast_input_dataframe(data, horizon, expected):
    '''
    test the correct number of error metric functions are returned.
    '''
    model = b.SNaive(1)
    model.fit(pd.DataFrame(data))
    #point forecasts only
    preds = model.predict(horizon)
    assert len(preds) == expected


@pytest.mark.parametrize("data, horizon, expected", 
                         [([1, 2, 3, 4, 5], 12, 12),
                          ([1, 2, 3, 4, 5], 24, 24),
                          ([1, 2, 3], 8, 8)
                          ])
def test_drift_forecast_input_numpy(data, horizon, expected):
    '''
    test the correct number of error metric functions are returned.
    '''
    model = b.Drift()
    model.fit(np.array(data))
    #point forecasts only
    preds = model.predict(horizon)
    assert len(preds) == expected



@pytest.mark.parametrize("data, horizon, expected", 
                         [([1, 2, 3, 4, 5], 12, 12),
                          ([1, 2, 3, 4, 5], 24, 24),
                          ([1, 2, 3], 8, 8)
                          ])
def test_drift_forecast_input_series(data, horizon, expected):
    '''
    test the correct number of error metric functions are returned.
    '''
    model = b.Drift()
    model.fit(pd.Series(data))
    #point forecasts only
    preds = model.predict(horizon)
    assert len(preds) == expected



@pytest.mark.parametrize("data, horizon, expected", 
                         [([1, 2, 3, 4, 5], 12, 12),
                          ([1, 2, 3, 4, 5], 24, 24),
                          ([1, 2, 3], 8, 8)
                          ])
def test_drift_forecast_input_dataframe(data, horizon, expected):
    '''
    test the correct number of error metric functions are returned.
    '''
    model = b.Drift()
    model.fit(pd.DataFrame(data))
    #point forecasts only
    preds = model.predict(horizon)
    assert len(preds) == expected



@pytest.mark.parametrize("data, expected", 
                         [([1, 2, 3, 4, 5], 3.0),
                          ([139,  32,  86, 123,  61,  51, 108, 137,  33,  25], 79.5),
                          ([1, 2, 3], 2.0)
                          ])
def test_average_forecast_output(data, expected):
    '''
    test the correct number of error metric functions are returned.
    '''
    model = b.Average()
    model.fit(pd.DataFrame(data))
    #point forecasts only
    preds = model.predict(1)
    assert preds[0] == expected


@pytest.mark.parametrize("data, expected", 
                         [([1, 2, 3, 4, 5], 5.0),
                          ([139,  32,  86, 123,  61,  51, 108, 137,  33,  25], 25.0),
                          ([1, 2, 3], 3.0)
                          ])
def test_naive1_forecast_output(data, expected):
    '''
    test naive1 carries forward the last value in the series
    '''
    model = b.Naive1()
    model.fit(pd.DataFrame(data))
    #point forecasts only
    preds = model.predict(1)
    assert preds[0] == expected


@pytest.mark.parametrize("data, exception", 
                         [(np.array([]), ValueError),
                          (1.0, TypeError),
                          (np.array(['foo', 'bar', 'spam', 'eggs']), TypeError),
                          (np.array([1, 2, 3, 4, 5, 6, np.NAN]), TypeError),
                          (np.array([1, 2, 3, 4, np.Inf, 5, 6]), TypeError)])
def test_naive1_abnormal_input(data, exception):
    '''
    test naive1 raises correct exceptions on abnormal input
    '''
    model = b.Naive1()
    with pytest.raises(exception):
        model.fit(data)

@pytest.mark.parametrize("data, exception", 
                         [(np.array([]), ValueError),
                          (1.0, TypeError),
                          (np.array(['foo', 'bar', 'spam', 'eggs']), TypeError),
                          (np.array([1, 2, 3, 4, 5, 6, np.nan]), TypeError),
                          (np.array([1, 2, 3, 4, np.Inf, 5, 6]), TypeError)
                          ])
def test_snaive_abnormal_input(data, exception):
    '''
    test snaive raises correct exceptions on abnormal input
    '''
    model = b.SNaive(1)
    with pytest.raises(exception):
        model.fit(data)

@pytest.mark.parametrize("data, exception", 
                         [(np.array([]), ValueError),
                          (1.0, TypeError),
                          (np.array(['foo', 'bar', 'spam', 'eggs']), TypeError),
                          (np.array([1, 2, 3, 4, 5, 6, np.nan]), TypeError),
                          (np.array([1, 2, 3, 4, np.Inf, 5, 6]), TypeError)])
def test_average_abnormal_input(data, exception):
    '''
    test average raises correct exceptions on abnormal input
    '''
    model = b.Average()
    with pytest.raises(exception):
        model.fit(data)

@pytest.mark.parametrize("data, exception", 
                         [(np.array([]), ValueError),
                          (1.0, TypeError),
                          (np.array(['foo', 'bar', 'spam', 'eggs']), TypeError),
                          (np.array([1, 2, 3, 4, 5, 6, np.nan]), TypeError),
                          (np.array([1, 2, 3, 4, np.Inf, 5, 6]), TypeError)])
def test_drift_abnormal_input(data, exception):
    '''
    test drift raises correct exceptions on abnormal input
    '''
    model = b.Drift()
    with pytest.raises(exception):
        model.fit(data)


@pytest.mark.parametrize("data, horizon, alphas, expected", 
                         [([1, 2, 3, 4, 5], 12, [0.2, 0.05], 12),
                          ([1, 2, 3, 4, 5], 24, [0.2, 0.10, 0.05], 24),
                          ([1, 2, 3], 8, [0.8], 8)
                          ])
def test_naive1_pi_horizon(data, horizon, alphas, expected):
    '''
    test the correct forecast horizon is returned for prediction interval
    for Naive1
    '''
    model = b.Naive1()
    model.fit(pd.Series(data))
    #point forecasts only
    preds, intervals = model.predict(horizon, return_predict_int=True, alphas=alphas)
    assert len(intervals[0]) == expected


@pytest.mark.parametrize("data, horizon, alphas, expected", 
                         [([1, 2, 3, 4, 5], 12, [0.2, 0.05], 12),
                          ([1, 2, 3, 4, 5], 24, [0.2, 0.10, 0.05], 24),
                          ([1, 2, 3], 8, [0.8], 8)
                          ])
def test_snaive_pi_horizon(data, horizon, alphas, expected):
    '''
    test the correct forecast horizon is returned for prediction interval for SNaive
    '''
    model = b.SNaive(1)
    model.fit(pd.Series(data))
    #point forecasts only
    preds, intervals = model.predict(horizon, return_predict_int=True, alphas=alphas)
    assert len(intervals[0]) == expected

@pytest.mark.parametrize("data, horizon, alphas, expected", 
                         [([1, 2, 3, 4, 5], 12, [0.2, 0.05], 12),
                          ([1, 2, 3, 4, 5], 24, [0.2, 0.10, 0.05], 24),
                          ([1, 2, 3], 8, [0.8], 8)
                          ])
def test_drift_pi_horizon(data, horizon, alphas, expected):
    '''
    test the correct forecast horizon is returned for prediction interval for Drift
    '''
    model = b.Drift()
    model.fit(pd.Series(data))
    #point forecasts only
    preds, intervals = model.predict(horizon, return_predict_int=True, alphas=alphas)
    assert len(intervals[0]) == expected

@pytest.mark.parametrize("data, horizon, alphas, expected", 
                         [([1, 2, 3, 4, 5], 12, [0.2, 0.05], 12),
                          ([1, 2, 3, 4, 5], 24, [0.2, 0.10, 0.05], 24),
                          ([1, 2, 3], 8, [0.8], 8)
                          ])
def test_average_pi_horizon(data, horizon, alphas, expected):
    '''
    test the correct forecast horizon is returned for prediction interval for Average
    '''
    model = b.Average()
    model.fit(pd.Series(data))
    #point forecasts only
    preds, intervals = model.predict(horizon, return_predict_int=True, alphas=alphas)
    assert len(intervals[0]) == expected

@pytest.mark.parametrize("model, data, horizon, alphas, expected", 
                         [(b.Naive1(), [1, 2, 3, 4, 5], 12, [0.2, 0.05], 2),
                          (b.Naive1(), [1, 2, 3, 4, 5], 24, [0.2, 0.10, 0.05], 3),
                          (b.SNaive(1), [1, 2, 3], 8, [0.8], 1),
                          (b.SNaive(1), [1, 2, 3, 4, 5], 24, [0.2, 0.10, 0.05], 3),
                          (b.Naive1(), [1, 2, 3], 8, None, 2),
                          (b.SNaive(1), [1, 2, 3], 8, None, 2),
                          (b.Average(), [1, 2, 3], 8, None, 2),
                          (b.Drift(), [1, 2, 3], 8, None, 2),
                          (b.Drift(), [1, 2, 3], 8, [0.8], 1),
                          (b.Drift(), [1, 2, 3], 8, None, 2),
                          (b.Average(), [1, 2, 3, 4, 5], 24, [0.2, 0.10, 0.05], 3)
                          ])
def test_naive_pi_set_number(model, data, horizon, alphas, expected):
    '''
    test the correct number of Prediction intervals are
    returned for prediction interval for all Naive forecasting classes
    '''
    model.fit(pd.Series(data))
    #point forecasts only
    preds, intervals = model.predict(horizon, return_predict_int=True, alphas=alphas)
    assert len(intervals) == expected

