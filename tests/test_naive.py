'''
Tests for Naive benchmark classes
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
def test_snaive_forecast_input_numpy(data, horizon, expected):
    '''
    test the correct number of error metric functions are returned.
    '''
    model = b.SNaive(1)
    model.fit(np.array(data))
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
    test the correct number of error metric functions are returned.
    '''
    model = b.Naive1()
    model.fit(pd.DataFrame(data))
    #point forecasts only
    preds = model.predict(1)
    assert preds[0] == expected