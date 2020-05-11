'''
Tests for Naive benchmark classes

Tests currently cover:

1. Forecast horizons
2. Allowable input types: np.ndarray, pd.DataFrame, pd.Series
3. Failure paths for abnormal input such as np.nan, non numeric, empty arrays and np.Inf
4. Predictions
    - naive1 - carries forward last value
    - snaive - carries forward previous h values
    - average - flat forecast of average
    - drift - previous value + gradient 

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


@pytest.mark.parametrize("data, period, expected", 
                         [(np.resize(np.arange(12), 24), 12, np.arange(12)),
                          (np.resize(np.arange(24), 48), 24, np.arange(24)),
                          (pd.Series(np.resize(np.arange(12), 24)), 12, pd.Series(np.arange(12)))
                          ])
def test_snaive_forecast_output(data, period, expected):
    '''
    test naive1 carries forward the last value in the series
    '''
    model = b.SNaive(period)
    model.fit(data)
    #point forecasts only
    preds = model.predict(period)
    assert np.array_equal(preds, expected)

@pytest.mark.parametrize("data, period, expected", 
                         [(np.resize(np.arange(12), 24), 12, np.full(12, np.arange(12).mean())),
                          (np.resize(np.arange(24), 48), 24, np.full(24, np.arange(24).mean())),
                          (pd.Series(np.resize(np.arange(12), 24)), 12, np.full(12, np.arange(12).mean()))
                          ])
def test_average_forecast_output_longer_horizon(data, period, expected):
    '''
    test naive1 carries forward the last value in the series
    '''
    model = b.Average()
    model.fit(data)
    #point forecasts only
    preds = model.predict(period)
    assert np.array_equal(preds, expected)


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


@pytest.mark.parametrize("data, period, expected", 
                         [(np.arange(1, 7), 6, np.arange(7, 13)),
                          (pd.Series(np.arange(1, 7)), 6, np.arange(7, 13)),
                          (pd.DataFrame(np.arange(1, 7)), 6, np.arange(7, 13)),
                          (pd.DataFrame(np.arange(1.0, 7.0, dtype=np.float64)), 6, 
                            np.arange(7.0, 13.0, dtype=np.float64))
                          ])
def test_drift_forecast_output_longer_horizon(data, period, expected):
    '''
    test drift forecast predictions
    '''
    model = b.Drift()
    model.fit(data)
    #point forecasts only
    preds = model.predict(period)
    assert np.array_equal(preds, expected)


def test_naive1_prediction_interval_low():
    '''
    test naive 80% lower prediction interval 
    '''

    np.random.seed(1066)
    train = np.random.poisson(lam=50, size=100)
    low = [29.56885, 24.005, 19.73657, 16.13770, 12.96704]
    high = [56.43115, 61.99451, 66.26343, 69.86230, 73.03296]

    model = b.Naive1()
    model.fit(train)
    preds, intervals = model.predict(5, return_predict_int=True, alphas=[0.2])
    print(intervals[0].T[0])
    assert pytest.approx(intervals[0].T[0], rel=1e-6, abs=0.1) == low


def test_naive1_prediction_interval_high():
    '''
    test naive 80% upper prediction interval 
    '''

    np.random.seed(1066)
    train = np.random.poisson(lam=50, size=100)
    low = [29.56885, 24.005, 19.73657, 16.13770, 12.96704]
    high = [56.43115, 61.99451, 66.26343, 69.86230, 73.03296]

    model = b.Naive1()
    model.fit(train)
    preds, intervals = model.predict(5, return_predict_int=True, alphas=[0.2])
    
    print(intervals[0].T[1])
    assert pytest.approx(intervals[0].T[1], rel=1e-6, abs=0.1) == high

def test_naive1_se():
    '''
    standard error of naive1 is root mean squared.
    '''
    np.random.seed(1066)
    train = np.random.poisson(lam=50, size=100)
    low = [29.56885, 24.005, 19.73657, 16.13770, 12.96704]
    high = [56.43115, 61.99451, 66.26343, 69.86230, 73.03296]

    model = b.Naive1()
    model.fit(train)

    expected = 10.48038

    assert pytest.approx(model._resid_std) == expected

def test_average_prediction_interval_high():
    '''
    test average 80% upper prediction interval 
    '''

    np.random.seed(1066)
    train = np.random.poisson(lam=50, size=100)
    low = [40.97369, 40.97369, 40.97369, 40.97369, 40.97369]
    high = [59.34631, 59.34631, 59.34631, 59.34631, 59.34631]

    model = b.Average()
    model.fit(train)
    preds, intervals = model.predict(5, return_predict_int=True, alphas=[0.2])
    
    print(intervals[0].T[1])
    #assert np.array_equal(intervals[0].T[1], high)
    assert pytest.approx(intervals[0].T[1]) == high



def test_average_prediction_interval_low():
    '''
    test average 80% lower prediction interval 
    '''
    np.random.seed(1066)
    train = np.random.poisson(lam=50, size=100)
    low = [40.97369, 40.97369, 40.97369, 40.97369, 40.97369]
    high = [59.34631, 59.34631, 59.34631, 59.34631, 59.34631]

    model = b.Average()
    model.fit(train)
    preds, intervals = model.predict(5, return_predict_int=True, alphas=[0.2])
    
    print(intervals[0].T[1])
    #assert np.array_equal(intervals[0].T[1], high)
    assert pytest.approx(intervals[0].T[0]) == low

def test_naive1_prediction_interval_95_high():
    '''
    test naive1 95% upper prediction interval 
    '''

    np.random.seed(1066)
    train = np.random.poisson(lam=50, size=100)
    low = [22.458831, 13.950400, 7.421651, 1.917662, -2.931450]
    high = [63.54117, 72.04960, 78.57835, 84.08234, 88.93145]

    model = b.Naive1()
    model.fit(train)
    preds, intervals = model.predict(5, return_predict_int=True, alphas=[0.05])
    
    print(intervals[0].T[1])
    assert pytest.approx(intervals[0].T[1], rel=1e-6, abs=0.1) == high

def test_naive1_prediction_interval_95_low():
    '''
    test naive1 95% lower prediction interval 
    '''

    np.random.seed(1066)
    train = np.random.poisson(lam=50, size=100)
    low = [22.458831, 13.950400, 7.421651, 1.917662, -2.931450]
    high = [63.54117, 72.04960, 78.57835, 84.08234, 88.93145]

    model = b.Naive1()
    model.fit(train)
    preds, intervals = model.predict(5, return_predict_int=True, alphas=[0.05])
    
    print(intervals[0].T[0])
    assert pytest.approx(intervals[0].T[0], rel=1e-6, abs=0.1) == low


def test_snaive_prediction_interval_80_low():
    '''
    test snaive 80% lower prediction interval 
    intervals are matched from R forecast package
    '''
    np.random.seed(1066)
    train = np.random.poisson(lam=50, size=100)
    low = [32.00420, 57.00420, 49.00420, 30.00420, 26.62116]
    high = [57.99580, 82.99580, 74.99580, 55.99580, 63.37884]

    #quarterly data
    model = b.SNaive(period=4)
    model.fit(train)
    preds, intervals = model.predict(5, return_predict_int=True, alphas=[0.2])
    
    print(intervals[0].T[0])
    assert pytest.approx(intervals[0].T[0]) == low

def test_snaive_prediction_interval_80_high():
    '''
    test snaive 80% upper prediction interval 
    intervals are matched from R forecast package
    '''
    np.random.seed(1066)
    train = np.random.poisson(lam=50, size=100)
    low = [32.00420, 57.00420, 49.00420, 30.00420, 26.62116]
    high = [57.99580, 82.99580, 74.99580, 55.99580, 63.37884]

    #quarterly data
    model = b.SNaive(period=4)
    model.fit(train)
    preds, intervals = model.predict(5, return_predict_int=True, alphas=[0.2])
    
    print(intervals[0].T[1])
    assert pytest.approx(intervals[0].T[1]) == high


def test_snaive_prediction_interval_95_high():
    '''
    test snaive 95% upper prediction interval 
    intervals are matched from R forecast package
    '''
    np.random.seed(1066)
    train = np.random.poisson(lam=50, size=100)
    low = [25.12464, 50.12464, 42.12464, 23.12464, 16.89199]
    high = [64.87536, 89.87536, 81.87536, 62.87536, 73.10801]

    #quarterly data
    model = b.SNaive(period=4)
    model.fit(train)
    preds, intervals = model.predict(5, return_predict_int=True, alphas=[0.05])
    
    print(intervals[0].T[1])
    assert pytest.approx(intervals[0].T[1]) == high

def test_snaive_prediction_interval_95_low():
    '''
    test snaive 95% lower prediction interval 
    intervals are matched from R forecast package
    '''
    np.random.seed(1066)
    train = np.random.poisson(lam=50, size=100)
    low = [25.12464, 50.12464, 42.12464, 23.12464, 16.89199]
    high = [64.87536, 89.87536, 81.87536, 62.87536, 73.10801]

    #quarterly data
    model = b.SNaive(period=4)
    model.fit(train)
    preds, intervals = model.predict(5, return_predict_int=True, alphas=[0.05])
    
    print(intervals[0].T[0])
    assert pytest.approx(intervals[0].T[0]) == low


