'''
model_selection module:

Provides:
Tools to support the selection of the best forecasting model

In particular tools that support cross validation.

1. rolling_forecast_origin
    A train-test split generator.  Starting from a specified origin the 
    generator returns (train, test) splits where the origin of test 
    incrementally moves forward in time.

2. sliding_window
    A train-test split generator.  Similar to rolling_forecast_origin.  
    However there is a 'window size' i.e. the training set is a fixed size.

3. cross_validation_score
    Uses a train, test split generator, a model and a forecast error metric to 
    return statistics on a models prediction performance.

'''

import numpy as np
from joblib import Parallel, delayed


def rolling_forecast_origin(train, min_train_size, horizon, step=1):
    '''
    Rolling forecast origin generator.

    Parameters:
    --------
    train: array-like
        training data for time series method

    min_train_size: int
        lookback - initial training size

    horizon: int 
        forecast horizon. 

    step: int, optional (default=1)
        step=1 means that a single additional data point is added to the time
        series.  increase step to run less splits.

    Returns:
        array-like, array-like

        split_training, split_validation, horizons
    '''

    for i in range(0, len(train) - min_train_size - horizon + 1, step):
        split_train = train[:min_train_size+i]
        split_val = train[min_train_size+i:min_train_size+i+horizon]
        yield split_train, split_val


def sliding_window(train, window_size, horizon, step=1):
    '''
    sliding window  generator.

    Parameters:
    --------
    train: array-like
        training data for time series method

    window_size: int
        lookback - how much data to include.

    horizon: int 
        forecast horizon. 

    step: int, optional (default=1)
        step=1 means that a single additional data point is added to the time
        series.  increase step to run less splits.

    Returns:
        array-like, array-like

        split_training, split_validation, horizons
    '''

    for i in range(0, len(train) - window_size - horizon + 1, step):
        split_train = train[i:window_size+i]
        split_val = train[i+window_size:window_size+i+horizon]
        yield split_train, split_val


def forecast_accuracy(model, train, test, horizons, metric):
    '''
    Returns forecast accuracy of model fit on 
    training data and compared against a test set 
    with a given metric.  

    Allows multiple forecast horizons.  The model predicts
    the maximum forecast horizon and then calculates the 
    accuracy across each.

    Parameters:
    ----------
    model - object
        forecasting model with .fit(train) and
        .predict(horizon) methods

    train - array-like
        training data

    test: array-like
        holdout data for testing

    horizons: list
        list of forecast horizons e.g. [7, 14, 28]

    metric: function
        error measure sig (y_true, y_preds)

    Returns:
    -------
    float
    '''
    h_accuracy = []
    model.fit(train)
    preds = model.predict(max(horizons))
    for horizon in horizons:
        score = metric(y_true=test[:horizon], y_pred=preds[:horizon])
        h_accuracy.append(score)
    return h_accuracy


def cross_validation_score(model, train, cv, metric, horizons=None, n_jobs=-1):
    '''
    Calculate cross validation scores

    Parameters:
    ----------
    model: object
        forecast model

    train: array-like
        training data

    metric: func(y_true, y_pred)
        forecast error metric

    horizons: list, optional (default=None):
        If the user wishes to return cross validation results
        for multiple sub-horizons e.g. within a 28 horizon [7, 14, 28]

    n_jobs: int, optional (default=-1)
        when -1 runs across all cores
        set = 1 to run each cross validation seperately.
        using -1 speeds up cross validation of slow running models.   

    Returns:
    -------
    array of arrays
    '''

    if horizons == None:
        cv_scores = Parallel(n_jobs=n_jobs)(delayed(forecast_accuracy)(model,
                                                                       cv_train,
                                                                       cv_test,
                                                                       [len(
                                                                           cv_test)],
                                                                       metric)
                                            for cv_train, cv_test in cv)
    else:

        cv_scores = Parallel(n_jobs=n_jobs)(delayed(forecast_accuracy)(model,
                                                                       cv_train,
                                                                       cv_test,
                                                                       horizons,
                                                                       metric)
                                            for cv_train, cv_test in cv)

    return np.array(cv_scores)
