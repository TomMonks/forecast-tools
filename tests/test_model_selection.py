'''
Tests for model selection / cross validation

At the moment these are all positive test.
This needs to be updated to include failure/exception testing.

'''


import pytest
import numpy as np
import pandas as pd

from forecast_tools import model_selection as ms
from forecast_tools import metrics 
from forecast_tools import baseline as b
                              
@pytest.mark.parametrize("train_size, horizon, expected", 
                         [(10, 1, 1),
                          (100, 1, 1),
                          (100, 10, 10),
                          (18, 6, 6),
                          (12, 3, 3)])
def test_rfo_test_length(train_size, horizon, expected):
    '''
    test the length of test is correct in the rolling origin
    generator
    '''
    train = np.arange(train_size)
    cv = ms.rolling_forecast_origin(train, min_train_size=1, horizon=horizon)

    _ , test_cv = next(cv)

    assert expected == len(test_cv)


@pytest.mark.parametrize("train_size, horizon, expected", 
                         [(10, 1, 1),
                          (100, 1, 1),
                          (100, 10, 10),
                          (18, 6, 6),
                          (12, 3, 3)])
def test_sw_test_length(train_size, horizon, expected):
    '''
    test the length of test is correct in the sliding window
    generator
    '''
    train = np.arange(train_size)
    cv = ms.sliding_window(train, window_size=1, horizon=horizon)

    _ , test_cv = next(cv)

    assert expected == len(test_cv)


@pytest.mark.parametrize("train_size, min_train_size, horizon, expected", 
                         [(10, 3, 1, 3),
                          (100, 34, 1, 34),
                          (100, 84, 10, 84),
                          (18, 12, 6, 12),
                          (12, 1, 3, 1)])
def test_rfo_min_train_length(train_size, min_train_size, horizon, expected):
    '''
    check that the minimum training size is correct
    '''
    train = np.arange(train_size)
    cv = ms.rolling_forecast_origin(train, min_train_size=min_train_size, 
                                    horizon=horizon)

    train_cv , _ = next(cv)

    assert expected == len(train_cv)


@pytest.mark.parametrize("train_size, window_size, horizon, expected", 
                         [(10, 3, 1, 3),
                          (100, 34, 1, 34),
                          (100, 84, 10, 84),
                          (18, 12, 6, 12),
                          (12, 1, 3, 1)])
def test_sw_first_window_size(train_size, window_size, horizon, expected):
    '''
    check that the window size is correct in the sliding
    window method - this tests the first fold.
    '''
    train = np.arange(train_size)
    cv = ms.sliding_window(train, window_size=window_size, 
                                    horizon=horizon)

    train_cv , _ = next(cv)

    assert expected == len(train_cv)


@pytest.mark.parametrize("train_size, window_size, horizon, expected", 
                         [(10, 3, 1, 3),
                          (100, 34, 1, 34),
                          (100, 84, 10, 84),
                          (18, 6, 6, 6),
                          (12, 1, 3, 1)])
def test_sw_second_window_size(train_size, window_size, horizon, expected):
    '''
    check that the window size is correct in the sliding
    window method - this tests the second fold.
    '''
    train = np.arange(train_size)
    cv = ms.sliding_window(train, window_size=window_size, 
                                    horizon=horizon)

    train_cv , _ = next(cv)
    #test the second fold
    train_cv , _ = next(cv)

    assert expected == len(train_cv)


@pytest.mark.parametrize("train_size, min_train_size, horizon, step, expected", 
                         [(10, 3, 1, 1, 4),
                          (100, 34, 1, 10, 44),
                          (100, 3, 10, 9, 12),
                          (10000, 999, 6, 34, 999+34),
                          (12, 1, 3, 5, 6)])
def test_rfo_second_fold_train_size(train_size, min_train_size, horizon, step,
                                    expected):
    '''
    check that the second fold size = minimum training size + step
    '''
    train = np.arange(train_size)
    cv = ms.rolling_forecast_origin(train, min_train_size=min_train_size, 
                                    horizon=horizon, step=step)

    train_cv , _ = next(cv)
    #second fold min_train_size + step == len
    train_cv , _ = next(cv)

    assert expected == len(train_cv)


@pytest.mark.parametrize("n_horizons, expected", 
                         [(1, 1),
                          (2, 2),
                          (3, 3),
                          (15, 15),
                          (45, 45)])
def test_forecast_accuracy_length(n_horizons, expected):
    model = b.Naive1()
    train = np.arange(10000)
    metric = metrics.mean_absolute_error
    horizons = np.arange(1, n_horizons+1).tolist()
    cv = ms.rolling_forecast_origin(train, min_train_size=100, 
                                    horizon=100)
    train_cv , test_cv = next(cv)

    result_h = ms.forecast_accuracy(model, train_cv, test_cv, 
                                    horizons=horizons, metric=metric)

    assert len(result_h) == expected


@pytest.mark.parametrize("train_size, min_train_size, horizon, step, expected", 
                         [(10, 3, 1, 1, 7),
                          (10, 3, 2, 1, 6),
                          (10, 3, 1, 2, 4),
                          (10, 3, 2, 2, 3)])
def test_rfo_number_of_folds(train_size, min_train_size, horizon, step,
                                    expected):
    '''
    check that the number of folds returned from rolling origin
    is as expected 
    '''
    train = np.arange(train_size)
    cv = ms.rolling_forecast_origin(train, min_train_size=min_train_size, 
                                    horizon=horizon, step=step)
    actual = 0

    #number of folds found
    for _ , _ in cv:
        actual += 1
    
    assert expected == actual


@pytest.mark.parametrize("train_size, window_size, horizon, step, expected", 
                         [(10, 3, 1, 1, 7),
                          (10, 3, 2, 1, 6),
                          (10, 3, 1, 2, 4),
                          (10, 3, 2, 2, 3)])
def test_sw_number_of_folds(train_size, window_size, horizon, step,
                                    expected):
    '''
    check that the number of folds returned from sliding window
    is as expected 
    '''
    train = np.arange(train_size)
    cv = ms.sliding_window(train, window_size=window_size, 
                                    horizon=horizon, step=step)
    actual = 0

    #number of folds found
    for _ , _ in cv:
        actual += 1
    
    assert expected == actual


@pytest.mark.parametrize("train_size, window_size, horizon, step, expected", 
                         [(10, 3, 1, 1, 7),
                          (10, 3, 2, 1, 6),
                          (10, 3, 1, 2, 4),
                          (10, 3, 2, 2, 3)])
def test_sw_number_of_folds_pd(train_size, window_size, horizon, step,
                                    expected):
    '''
    check that the number of folds returned from rolling origin
    is as expected when data source is a PANDAS.DATAFRAME
    '''
    train = pd.DataFrame(np.arange(train_size))
    cv = ms.sliding_window(train, window_size=window_size, 
                                    horizon=horizon, step=step)
    actual = 0

    #number of folds found
    for _ , _ in cv:
        actual += 1
    
    assert expected == actual


@pytest.mark.parametrize("train_size, min_train_size, horizon, step, expected", 
                         [(10, 3, 1, 1, 7),
                          (10, 3, 2, 1, 6),
                          (10, 3, 1, 2, 4),
                          (10, 3, 2, 2, 3)])
def test_rfo_number_of_folds_pd(train_size, min_train_size, horizon, step,
                                    expected):
    '''
    check that the number of folds returned from rolling origin
    is as expected when data source is a pandas.DataFrame
    '''
    train = pd.DataFrame(np.arange(train_size))
    cv = ms.rolling_forecast_origin(train, min_train_size=min_train_size, 
                                    horizon=horizon, step=step)
    actual = 0

    #number of folds found
    for _ , _ in cv:
        actual += 1
    
    assert expected == actual

