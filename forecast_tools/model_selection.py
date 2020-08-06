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

Dev notes:
--------

At the moment coverage is not included.  I have handled this in practical
projects by including a function that returns all of the predictions within a
split/fold. A seperate function then handles coverage.
'''

import numpy as np
from joblib import Parallel, delayed

from forecast_tools.metrics import (mean_absolute_scaled_error,
                                    _forecast_error_functions)
from forecast_tools.baseline import baseline_estimators


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
    Forecast accuracy of a model over multiple horizons

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


def cross_validation_score(model, cv, metric, horizons=None,
                           n_jobs=-1):
    '''
    Cross validation scores

    Parameters:
    ----------
    model: object
        forecast model

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

    if horizons is None:
        cv_scores = \
            Parallel(n_jobs=n_jobs)(delayed(forecast_accuracy)(model,
                                                               cv_train,
                                                               cv_test,
                                                               [len(cv_test)],
                                                               metric)
                                    for cv_train, cv_test in cv)
    else:

        cv_scores = \
            Parallel(n_jobs=n_jobs)(delayed(forecast_accuracy)(model,
                                                               cv_train,
                                                               cv_test,
                                                               horizons,
                                                               metric)
                                    for cv_train, cv_test in cv)

    return np.array(cv_scores)


def forecast(model, train, test, horizon):
    '''
    h-step prediction of a model

    Returns a tuple of (y_preds, y_train, y_true) of model fit
    to training data

    Parameters:
    ----------
    model - object
        forecasting model with .fit(train) and
        .predict(horizon) methods

    train - array-like
        training data

    test: array-like
        holdout data for testing

    horizon: int
        forecast horizon

    Returns:
    --------
    tuple (y_pred, y_train, y_true)
    '''
    y_pred = model.fit_predict(train, horizon)
    return train, test, y_pred


def cross_validation_folds(model, cv, n_jobs=-1):
    '''
    Cross validation forecasts

    Parameters:
    ----------
    model: object
        forecast model

    cv: object
        cross validation generator
        i.e. rolling_forecast_origin or sliding_window

    n_jobs: int, optional (default=-1)
        when -1 runs across all cores
        set = 1 to run each cross validation seperately.
        using -1 speeds up cross validation of slow running models.

    Returns:
    -------
    np.ndarray of tuples
    each tuple is (cv_train, cv_test, cv_y_pred)
    '''

    cv_folds = \
        Parallel(n_jobs=n_jobs)(delayed(forecast)(model,
                                                  cv_train,
                                                  cv_test,
                                                  len(cv_test))
                                for cv_train, cv_test in cv)

    return np.array(cv_folds)


def scaled_cross_validation_score(model, cv, seasonal_period=None):
    '''
    Mean absolute scaled error cross validation score

    Parameters:
    ----------
    model: object
        forecast model

    cv: generator
        time series cross validation fold generator

    metric: func(y_true, y_py_red)
        forecast error metric

    seasonal_period: None or int, optional (default=None)
        if none the in-sample one step Naive1 used for scaling.
        if int SNaive is used instead.

    Returns:
    --------
    np.array of mase scores
    '''

    folds = cross_validation_folds(model, cv)

    scores = []
    for y_train, y_true, y_pred in folds:
        score = mean_absolute_scaled_error(y_true, y_pred, y_train,
                                           period=seasonal_period)
        scores.append(score)

    return np.array(scores)


def auto_naive(y_train, horizon=1, seasonal_period=1, min_train_size='auto',
               method='cv', step=1, window_size='auto', metric='mae'):
    '''Automatic selection of the 'best' naive benchmark on a 'single' series

    The selection process uses out-of-sample cv performance.

    By default auto_naive uses cross validation to estimate the mean
    point forecast peformance of all naive methods.  It selects the method
    with the lowest point forecast metric on average.

    If there is limited data for training a basic holdout sample could be
    used.

    Dev note: the plan is to update this to work with multiple series.
    It would be best to use MASE for multiple series comparison.

    Parameters:
    ----------
    y_train: array-like
        training data.  typically in a pandas.Series, pandas.DataFrame
        or numpy.ndarray format.

    horizon: int, optional (default=1)
        Forecast horizon.

    seasonal_period: int, optional (default=1)
        Frequency of the data.  E.g. 7 for weekly pattern, 12 for monthly
        365 for daily.

    min_train_size: int or str, optional (default='auto')
        The size of the initial training set (if method=='ro' or 'sw').
        If 'auto' then then min_train_size is set to len(y_train) // 3
        If main_train_size='auto' and method='holdout' then
        min_train_size = len(y_train) - horizon.

    method: str, optional (default='cv')
        out of sample selection method.
        'ro' - rolling forecast origin
        'sw' - sliding window
        'cv' - scores from both ro and sw
        'holdout' - single train/test split
         Methods'ro' and 'sw' are similar, however, sw has a fixed
         window_size and drops older data from training.

    step: int, optional (default=1)
        The stride/step of the cross-validation. I.e. the number
        of observations to move forward between folds.

    window_size: str or int, optional (default='auto')
        The window_size if using sliding window cross validation
        When 'auto' and method='sw' then
        window_size=len(y_train) // 3

    metric: str, optional (default='mae')
        The metric to measure out of sample accuracy.
        Options: mase, mae, mape, smape, mse, rmse, me.

    Returns:
    --------
    dict
        'model': baseline.Forecast
        f'{metric}': float

        Contains the model and its CV performance.

    Raises:
    -------
    ValueError
        For invalid method, metric, window_size parameters

    See Also:
    --------
    forecast_tools.baseline.Naive1
    forecast_tools.baseline.SNaive
    forecast_tools.baseline.Drift
    forecast_tools.baseline.Average
    forecast_tools.baseline.EnsembleNaive
    forecast_tools.baseline.baseline_estimators
    forecast_tools.model_selection.rolling_forecast_origin
    forecast_tools.model_selection.sliding_window
    forecast_tools.model_selection.mase_cross_validation_score
    forecast_tools.metrics.mean_absolute_scaled_error

    Examples:
    ---------
    Measuring MAE and taking the best method using both
    rolling origin and sliding window cross validation
    of a 56 day forecast.

    >>> from forecast_tools.datasets import load_emergency_dept
    >>> y_train = load_emergency_dept
    >>> best = auto_naive(y_train, seasonal_period=7, horizon=56)
    >>> best
    {'model': Average(), 'mae': 19.63791579700355}


    Take a step of 7 days between cv folds.

    >>> from forecast_tools.datasets import load_emergency_dept
    >>> y_train = load_emergency_dept
    >>> best = auto_naive(y_train, seasonal_period=7, horizon=56,
        ...               step=7)
    >>> best
    {'model': Average(), 'mae': 19.675635558539383}

    '''
    valid_methods = ['holdout', 'ro', 'sw', 'cv']
    metrics = _forecast_error_functions()
    metrics['mase'] = mean_absolute_scaled_error

    if method not in valid_methods:
        raise ValueError(f"Method must be in {valid_methods}")

    if metric not in metrics:
        raise ValueError(f"Please select a metric from {list(metrics.keys())}")

    if min_train_size == 'auto':
        min_train_size = len(y_train) // 3
    elif not type(min_train_size) is int:
        raise ValueError(f"valid min_train_size values are 'auto' or int > 0")
    elif min_train_size < 1:
        raise ValueError(f"valid min_train_size values are 'auto' or int > 0")

    if window_size == 'auto':
        window_size = len(y_train) // 3
    elif not type(window_size) is int:
        raise ValueError(f"valid window_size values are 'auto' or int > 0")
    elif window_size < 1:
        raise ValueError(f"valid window_size values are 'auto' or int > 0")

    baselines = baseline_estimators(seasonal_period)

    method_score = []
    if method == 'cv':
        for _, model in baselines.items():
            cv_ro = rolling_forecast_origin(train=y_train,
                                            min_train_size=min_train_size,
                                            horizon=horizon,
                                            step=step)

            cv_sw = sliding_window(train=y_train,
                                   window_size=window_size,
                                   horizon=horizon,
                                   step=step)

            if metric == 'mase':
                score_ro = scaled_cross_validation_score(model, cv_ro,
                                                         seasonal_period)
                score_sw = scaled_cross_validation_score(model, cv_sw,
                                                         seasonal_period)

            else:
                score_ro = cross_validation_score(model, cv_ro,
                                                  metrics[metric])
                score_sw = cross_validation_score(model, cv_sw,
                                                  metrics[metric])

            score = np.concatenate([score_ro, score_sw])
            method_score.append(score.mean())

    elif method == 'ro':
        for _, model in baselines.items():
            cv = rolling_forecast_origin(train=y_train,
                                         min_train_size=min_train_size,
                                         horizon=horizon,
                                         step=step)

            if metric == 'mase':
                score_ro = scaled_cross_validation_score(model, cv,
                                                         seasonal_period)

            else:
                score_ro = cross_validation_score(model, cv, metrics[metric])

            method_score.append(score_ro.mean())

    elif method == 'sw':
        for _, model in baselines.items():
            cv = sliding_window(train=y_train,
                                window_size=window_size,
                                horizon=horizon,
                                step=step)

            if metric == 'mase':
                score_sw = scaled_cross_validation_score(model, cv,
                                                         seasonal_period)

            else:
                score_sw = cross_validation_score(model, cv, metrics[metric])

            method_score.append(score_sw.mean())

    else:
        # single train test split
        min_train_size = len(y_train) - horizon
        train = y_train[:min_train_size]
        test = y_train[min_train_size:]

        for _, model in baselines.items():
            model.fit(train)
            y_preds = model.predict(horizon)
            if metric == 'mase':
                score = metrics[metric](test, y_preds, y_train,
                                        seasonal_period)
            else:
                score = metrics[metric](test, y_preds)

            method_score.append(score.mean())

    method_score = np.array(method_score)
    best_index = np.argmin(method_score)

    best = {'model': list(baselines.items())[best_index][1],
            f'{metric}': method_score[best_index]}

    return best
