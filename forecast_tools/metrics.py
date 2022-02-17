'''
Metrics to measure forecast error

ME - mean error
MAE - mean absolute error
MSE - mean squared error
RMSE - root mean squared error
MAPE - mean absolute percentage error
sMAPE - symmetric MAPE.
MASE - mean absolute scaled error

coverage - prediction interval coverage
'''
import numpy as np
import pandas as pd

from forecast_tools.baseline import SNaive


def as_arrays(y_true, y_pred):
    '''
    Returns ground truth and predict
    values as numpy arrays.

    Parameters:
    --------
    y_true -- array-like
        actual observations from time series
    y_pred -- array-like
        the predictions

    Returns:
    -------
    Tuple(np.array np.array)
    '''
    return np.asarray(y_true), np.asarray(y_pred)


def mean_error(y_true, y_pred):
    '''
    Computes Mean Error (ME).

    Parameters:
    --------
    y_true -- array-like
        actual observations from time series
    y_pred -- arraylike
        the predictions to evaluate

    Returns:
    -------
    float
        scalar value representing the ME
    '''
    y_true, y_pred = as_arrays(y_true, y_pred)
    return np.mean(y_true - y_pred)


def mean_absolute_percentage_error(y_true, y_pred):
    '''
    Mean Absolute Percentage Error (MAPE).

    MAPE is a relative error measure of forecast accuracy.

    Limitations of MAPE ->

    1. When the ground true value is close to zero MAPE is inflated.

    2. MAPE is not symmetric.  MAPE produces smaller forecast
    errors when underforecasting.

    Parameters:
    --------
    y_true -- array-like
        actual observations from time series
    y_pred -- arraylike
        the predictions to evaluate

    Returns:
    -------
    float,
        scalar value representing the MAPE (0-100)
    '''
    y_true, y_pred = as_arrays(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def mean_absolute_error(y_true, y_pred):
    '''
    Mean Absolute Error (MAE)

    Parameters:
    --------
    y_true -- array-like
        actual observations from time series
    y_pred -- arraylike
        the predictions to evaluate

    Returns:
    -------
    float,
        scalar value representing the MAE
    '''
    y_true, y_pred = as_arrays(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred)))


def mean_squared_error(y_true, y_pred):
    '''
    Mean Squared Error (MSE)

    Parameters:
    --------
    y_true -- array-like
        actual observations from time series
    y_pred -- arraylike
        the predictions to evaluate

    Returns:
    -------
    float,
        scalar value representing the MSE
    '''
    y_true, y_pred = as_arrays(y_true, y_pred)
    return np.mean(np.square((y_true - y_pred)))


def root_mean_squared_error(y_true, y_pred):
    '''
    Root Mean Squared Error (RMSE).

    Square root of the mean squared error.

    Parameters:
    --------
    y_true -- array-like
        actual observations from time series
    y_pred -- arraylike
        the predictions to evaluate

    Returns:
    -------
    float,
        scalar value representing the RMSE
    '''
    y_true, y_pred = as_arrays(y_true, y_pred)
    return np.sqrt(mean_squared_error(y_true, y_pred))


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    '''
    Symmetric Mean Absolute Percentage Error (sMAPE)

    A proposed improvement./replacement for MAPE.  (But still not symmetric).

    Computation based on Hyndsight blog:
    https://robjhyndman.com/hyndsight/smape/

    Limitations of sMAPE:

    1. When the ground true value is close to zero MAPE is inflated.
    2. Like MAPE it is not symmetric.

    Parameters:
    --------
    y_true -- array-like
        actual observations from time series
    y_pred -- arraylike
        the predictions to evaluate

    Returns:
    -------
    float,
        scalar value representing the RMSE
    '''
    y_true, y_pred = as_arrays(y_true, y_pred)
    numerator = 2 * np.abs(y_true - y_pred)
    denominator = np.abs(y_pred) + np.abs(y_true)
    return np.mean(100 * (numerator / denominator))


def mean_absolute_scaled_error(y_true, y_pred, y_train, period=None):
    '''
    Mean absolute scaled error (MASE)

    MASE = MAE / MAE_{insample, naive}

    For definition: https://otexts.com/fpp2/accuracy.html

    Parameters:
    --------
    y_true: array-like
        actual observations from time series

    y_pred: array-like
        the predictions to evaluate

    y_train: array-like
        the training data the produced the predictions

    period: int or None, optional (default = None)
        if None then out of sample MAE is scaled by 1-step in-sample Naive1
        MAE.  If = int then SNaive is used as the scaler.

    Returns:
    -------
    float,
        scalar value representing the MASE
    '''
    y_true, y_pred = as_arrays(y_true, y_pred)

    if period is None:
        period = 1

    in_sample = SNaive(period=period)
    in_sample.fit(y_train)

    mae_insample = mean_absolute_error(y_train[period:],
                                       in_sample.fittedvalues.dropna())

    return mean_absolute_error(y_true, y_pred) / mae_insample


def forecast_errors(y_true, y_pred, metrics='all'):
    '''
    Convenience function for return a multiple
    forecast errors

    Parameters:
    --------
    y_true -- array-like
        actual observations from time series

    y_pred -- array-like
        the predictions to evaluate

    metrics -- str or List
        forecast error metrics to compute.
        'all' returns all forecast errors available
        List options: ['me', 'mae', 'mse', 'rmse', 'mape', 'smape']

    Returns:
    -------
    dict,
        forecast error metrics

    Example:
    ---------
    >>> y_true = [45, 60, 23, 45]
    >>> y_preds = [50, 50, 50, 50]

    >>> metrics = forecast_errors(y_true, y_preds)
    >>> print(metrics)

    >>> metrics = forecast_errors(y_true, y_preds, metrics=['mape', 'smape'])
    >>> print(metrics)

    '''
    y_true, y_pred = as_arrays(y_true, y_pred)

    if metrics == 'all':
        metrics = ['me', 'mae', 'mse', 'rmse', 'mape', 'smape']

    funcs = _forecast_error_functions()
    errors = {}
    for metric in metrics:
        errors[metric] = funcs[metric](y_true, y_pred)

    return errors


def _forecast_error_functions():
    '''
    Return all forecast functions in
    a dict

    Returns:
    --------
        dict
    '''
    funcs = {}
    funcs['me'] = mean_error
    funcs['mae'] = mean_absolute_error
    funcs['mse'] = mean_squared_error
    funcs['rmse'] = root_mean_squared_error
    funcs['mape'] = mean_absolute_percentage_error
    funcs['smape'] = symmetric_mean_absolute_percentage_error
    return funcs


def coverage(y_true, pred_intervals):
    '''
    Prediction Interval Coverage

    Calculates the proportion of the true
    values are that are covered by the lower
    and upper bounds of the prediction intervals

    Parameters:
    -------
    y_true -- array-like,
        actual observations

    pred_intervals -- np.array, matrix (hx2)
        prediction intervals

    Returns:
    -------
    float
    '''
    y_true = np.asarray(y_true)
    lower = np.asarray(pred_intervals.T[0])
    upper = np.asarray(pred_intervals.T[1])

    cover = len(np.where((y_true > lower) & (y_true < upper))[0])
    return cover / len(y_true)


def winkler_score(intervals, observations, alpha):
    '''
    Returns the mean winkler score of a set of observations and prediction
    intervals

    A Winkler score is the width of the interval plus a penality proportional
    to the deviation (above or below the interval) and 2/$\alpha$

    Smaller winkler scores are better.

    Parameters:
    -----------
    intervals: array-like
        array of prediction intervals

    observations: float or array-like
        individual observation or array of ground truth observations

    alpha: float
        The prediction interval alpha.  For an 80% pred intervals alpha=0.2

    Returns:
    -------
    float, numpy.ndarray

    Example usage:
    --------------

    Individual winkler score:
    ```python
    >>> alpha = 0.2
    >>> interval = [744.54, 773.22]
    >>> y_t = 741.84
    >>> ws = winkler_score(interval, y_t, alpha)
    >>> print(round(ws, 2))

    56.68
    ```

    Multiple interval scores

    ```python
    >>> TARGET = 0.80
    >>> HOLDOUT = 14
    >>> PERIOD = 7
    >>>
    >>> attends = load_emergency_dept()
    >>> # train-test split
    >>> train, test = attends[:-HOLDOUT], attends[-HOLDOUT:]
    >>> model = SNaive(PERIOD)
    >>> # returns 80 and 90% prediction intervals by default.
    >>> preds, intervals_ed = model.fit_predict(train, HOLDOUT,
        ... return_predict_int=True)
    >>> ws = mean_winkler_score_np(intervals_ed[0], test_ed, alpha=1-TARGET)
    >>> print(f'Mean winkler score: {ws:.2f}')

    Mean winkler score: 79.72
    ```

    '''
    # distinguish between handling individual obs and multiple obs
    if isinstance(observations, (np.ndarray, pd.DataFrame, list)):
        if len(observations) > 1:
            observations = np.array(observations).T[0]
        else:
            observations = np.array(observations)
    else:
        observations = np.array([observations])

    # handle intervals for an individual observation
    if len(intervals) == 2:
        intervals = np.array(intervals).reshape(1, -1)

    # interval widths
    scores = intervals[:, 1] - intervals[:, 0]

    # observation falls below lower interval
    below_lower = observations < intervals[:, 0]

    # observation exceeds upper interval
    above_upper = observations > intervals[:, 1]

    # lower penality
    scores[below_lower] += \
        ((2/alpha) * (intervals[:, 0]
         [below_lower] - observations[below_lower]))

    # upper penality
    scores[above_upper] += \
        ((2/alpha) * (observations[above_upper] -
         intervals[:, 1][above_upper]))

    return scores.mean()


if __name__ == '__main__':
    y_true = [45, 60, 23, 45]
    y_preds = [50, 50, 50, 50]

    metrics = forecast_errors(y_true, y_preds)
    print(metrics)

    metrics = forecast_errors(y_true, y_preds, metrics=['mape', 'smape'])
    print(metrics)
