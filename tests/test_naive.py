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
    test the correct number of error metric functions are returned.
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
def test_average_forecast_input_dataframe(data, horizon, expected):
    '''
    test the correct number of error metric functions are returned.
    '''
    model = b.Average()
    model.fit(pd.DataFrame(data))
    #point forecasts only
    preds = model.predict(horizon)
    assert len(preds) == expected