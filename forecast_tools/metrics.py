"""
Metrics to measure forecast error

ME - mean error
MAE - mean absolute error
MSE - mean squared error
RMSE - root mean squared error
MAPE - mean absolute percentage error
sMAPE - symmetric MAPE.
MASE - mean absolute scaled error

coverage - prediction interval coverage
winkler_score - winkler score for prediction interval
absolute_coverage_difference - difference of coverage from target
"""

import numpy as np
import pandas as pd
import numbers

import numpy.typing as npt
from typing import Optional, Dict, Tuple, List, Union

from forecast_tools.baseline import SNaive


def as_arrays(y_true, y_pred):
    """
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
    """
    return np.asarray(y_true), np.asarray(y_pred)


def mean_error(y_true, y_pred):
    """
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
    """
    y_true, y_pred = as_arrays(y_true, y_pred)
    return np.mean(y_true - y_pred)


def mean_absolute_percentage_error(y_true, y_pred):
    """
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
    """
    y_true, y_pred = as_arrays(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def mean_absolute_error(y_true, y_pred):
    """
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
    """
    y_true, y_pred = as_arrays(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred)))


def mean_squared_error(y_true, y_pred):
    """
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
    """
    y_true, y_pred = as_arrays(y_true, y_pred)
    return np.mean(np.square((y_true - y_pred)))


def root_mean_squared_error(y_true, y_pred):
    """
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
    """
    y_true, y_pred = as_arrays(y_true, y_pred)
    return np.sqrt(mean_squared_error(y_true, y_pred))


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """
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
    """
    y_true, y_pred = as_arrays(y_true, y_pred)
    numerator = 2 * np.abs(y_true - y_pred)
    denominator = np.abs(y_pred) + np.abs(y_true)
    return np.mean(100 * (numerator / denominator))


def mean_absolute_scaled_error(y_true, y_pred, y_train, period=None):
    """
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
    """
    y_true, y_pred = as_arrays(y_true, y_pred)

    if period is None:
        period = 1

    in_sample = SNaive(period=period)
    in_sample.fit(y_train)

    mae_insample = mean_absolute_error(
        y_train[period:], in_sample.fittedvalues.dropna()
    )

    return mean_absolute_error(y_true, y_pred) / mae_insample


def forecast_errors(y_true, y_pred, metrics="all"):
    """
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

    """
    y_true, y_pred = as_arrays(y_true, y_pred)

    if metrics == "all":
        metrics = ["me", "mae", "mse", "rmse", "mape", "smape"]

    funcs = _forecast_error_functions()
    errors = {}
    for metric in metrics:
        errors[metric] = funcs[metric](y_true, y_pred)

    return errors


def _forecast_error_functions():
    """
    Return all forecast functions in
    a dict

    Returns:
    --------
        dict
    """
    funcs = {}
    funcs["me"] = mean_error
    funcs["mae"] = mean_absolute_error
    funcs["mse"] = mean_squared_error
    funcs["rmse"] = root_mean_squared_error
    funcs["mape"] = mean_absolute_percentage_error
    funcs["smape"] = symmetric_mean_absolute_percentage_error
    return funcs


def _validate_interval_inputs(
    y_true: npt.ArrayLike,
    intervals: npt.ArrayLike,
    alpha: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validates inputs for prediction interval evaluation functions.
    
    Parameters:
    -----------
    y_true: float, integer or array-like
        Individual observation or array of ground truth observations.
    
    intervals: array-like
        Array of prediction intervals.
        
    alpha: float, optional (default=None)
        The prediction interval alpha. For an 80% prediction interval, alpha=0.2.
        If set to None then alpha is not checked.

    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        Validated and properly formatted y_true and intervals arrays.
        
    Raises:
    -------
    ValueError
        If inputs don't meet validation requirements.
    TypeError
        If inputs are not of compatible types.
    """
    # Validate alpha
    if alpha is not None:
        if not 0 < alpha < 1:
            raise ValueError("Alpha must be between 0 and 1")
        
    # Check if y_true is of a compatible type before conversion
    if not (isinstance(y_true, (list, tuple, np.ndarray, pd.Series, pd.DataFrame)) or 
            isinstance(y_true, (int, float, complex, bool, np.number))):
        raise TypeError(f"y_true must be array-like or a number, got {type(y_true).__name__}")
    
    # Check if intervals is of a compatible type before conversion
    if not isinstance(intervals, (list, tuple, np.ndarray, pd.Series, pd.DataFrame)):
        raise TypeError(f"intervals must be array-like, got {type(intervals).__name__}")
    
    # Convert observations to numpy array
    if isinstance(y_true, (pd.Series, pd.DataFrame)):
        y_true = y_true.to_numpy().flatten()
    elif isinstance(y_true, list):
        y_true = np.array(y_true)
    elif isinstance(y_true, numbers.Number):
        # individual number
        y_true = np.array([y_true])
    else:
        y_true = np.asarray(y_true).flatten()
    
    # Handle intervals for an individual observation
    intervals = np.asarray(intervals)
    if len(intervals) == 2 and intervals.ndim == 1:
        intervals = np.array(intervals).reshape(1, -1)
    
    if len(intervals) == 0:
        raise ValueError("Intervals array is empty!")
    
    # Validate intervals shape
    if intervals.shape[1] != 2:
        raise ValueError(f"Each interval must have a lower and upper bound {intervals.shape}")
    
    # Validate interval bounds
    if np.any(intervals[:, 0] > intervals[:, 1]):
        raise ValueError("Lower bounds must be less than or equal to upper bounds")
    
    # Ensure matching dimensions
    if len(intervals) != len(y_true):
        raise ValueError("Number of intervals must match number of observations")
    
    return y_true, intervals


def coverage(
    y_true: npt.ArrayLike, 
    pred_intervals: npt.ArrayLike
) -> float:
    """
    Prediction Interval Coverage

    Calculates the proportion of the true values that are covered by the 
    prediction intervals (lower and upper bounds), including values exactly 
    on the boundaries.

    Parameters:
    -----------
    y_true: array-like
        Actual observations.

    pred_intervals: array-like
        Prediction intervals, where each interval is [lower, upper].

    Returns:
    --------
    float
        The proportion of observations that fall within the prediction intervals.
        
    Examples:
    ---------
    ```
    intervals = np.array(
        [
            [37520, 58225],
            [29059, 49764],
            [47325, 68030],
            [36432, 57137],
            [35865, 56570],
            [33419, 54124],
        ]
    )

    y_true = np.array([37463, 40828, 56148, 45342, 43741, 45907])
    mean_cov = coverage(y_true, intervals)
    print(round(mean_cov, 2))
    ```
    """
    # Validate inputs without requiring alpha
    y_true, pred_intervals = _validate_interval_inputs(y_true, pred_intervals)
    
    # Extract lower and upper bounds
    lower = pred_intervals[:, 0]
    upper = pred_intervals[:, 1]
    
    # Calculate coverage, including boundary values
    covered = np.sum((y_true >= lower) & (y_true <= upper))
    
    return covered / len(y_true)


def winkler_score(
    y_true: npt.ArrayLike,
    intervals: npt.ArrayLike,
    alpha: float,
    return_scores: Optional[bool] = False,
) -> Union[float, Dict]:
    """
    Returns the mean winkler score of a set of observations and prediction
    intervals

    A Winkler score is the width of the interval plus a penality proportional
    to the deviation (above or below the interval) and 2/$\alpha$

    Smaller winkler scores are better.

    Parameters:
    -----------
    y_true: float, integer or array-like
        individual observation or array of ground truth observations.
    
    intervals: array-like
        array of prediction intervals

    alpha: float
        The prediction interval alpha.  For an 80% pred intervals alpha=0.2

    return_scores: bool. optipnal (Default = False)
        Returns a dictionary containined both the mean winkler score and the
        individual scores

    Returns:
    -------
    float | dict
        mean winkler score or dict containing mean and individual scores

    Example usage:
    --------------

    Individual winkler score:
    ```python
    >>> alpha = 0.2
    >>> interval = [744.54, 773.22]
    >>> y_t = 741.84
    >>> ws = winkler_score(y_t, interval, alpha)
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
    >>> ws = winkler_score_np(test_ed, intervals_ed[0],, alpha=1-TARGET)
    >>> print(f'Mean winkler score: {ws:.2f}')

    Mean winkler score: 79.72
    ```

    """
    
    # validate inputs
    y_true, intervals = _validate_interval_inputs(y_true, intervals, alpha)

    # Vectorized calculation
    # interval widths
    widths = intervals[:, 1] - intervals[:, 0]
    below_mask = y_true < intervals[:, 0]
    above_mask = y_true > intervals[:, 1]

    # Calculate penalties
    penalty_factor = 2.0 / alpha
    penalty_below = np.zeros_like(widths)
    penalty_above = np.zeros_like(widths)

    if np.any(below_mask):
        penalty_below[below_mask] = penalty_factor * (
            intervals[below_mask, 0] - y_true[below_mask]
        )

    if np.any(above_mask):
        penalty_above[above_mask] = penalty_factor * (
            y_true[above_mask] - intervals[above_mask, 1]
        )

    scores = widths + penalty_below + penalty_above

    if return_scores:
        return {
            "mean_score": scores.mean(),
            "individual_scores": scores,
            "below_count": np.sum(below_mask),
            "above_count": np.sum(above_mask),
        }

    return scores.mean()


def absolute_coverage_difference(
        y_true: npt.ArrayLike, 
        intervals: npt.ArrayLike, 
        alpha: Optional[float] = 0.05
) -> float:
    """
    The absolute coverage difference (ACD)

    ACD is the absolute difference between the average coverage
    of a method and the desired empirical coverage (default = 95%).

    If the future values are outside the prediction intervals
    by a method an average of 2% of the time (coverage of 98%),
    ACD = |0.98 - 0.95| = 0.03

    Parameters:
    ------
    y_true: array-like
        The ground truth future values

    intervals: array-like
        The generated prediction intervals. Where
        len(pred_intervals) == len(y_true)

    alpha: float, optional (default = 0.05)
        The alpha used to generate the prediction intervals.
        E.g. 0.05 expects a 95% coverage.

    Returns:
    --------
    float

    Sources:
    --------
    M4 competition paper:
    https://www.sciencedirect.com/science/article/pii/S0169207019301128

    Examples:
    ---------
    ```python
    >>> intervals = np.array([[37520, 58225],
    ...                       [29059, 49764],
    ...                       [47325, 68030],
    ...                       [36432, 57137],
    ...                       [35865, 56570],
    ...                       [33419, 54124]])

    >>> y_true = np.array([37463, 40828, 56148,
    ...                    45342, 43741, 45907])

    >>> acd = absolute_coverage_difference(y_true, intervals,
    ...                                    alpha=0.05)
    >>> print(round(acd, 2))

    0.12
    ```
    """

    # validate inputs
    y_true, intervals = _validate_interval_inputs(y_true, intervals, alpha)

    mean_coverage = coverage(y_true, intervals)
    return abs(mean_coverage - (1-alpha))


def interval_scores(
        y_true: npt.ArrayLike, 
        intervals: npt.ArrayLike, 
        alpha: float, 
        metrics: Union[str, List[str]] = "all"
) -> Dict[str, float]:
    """
    Convenience function for return a multiple
    interval evaluation scores.

    Parameters:
    --------
    y_true: array-like
        The ground truth future values

    intervals: array-like
        The generated prediction intervals. Where
        len(pred_intervals) == len(y_true)

    alpha: float
        The alpha used to generate the prediction intervals.
        E.g. 0.05 expects a 95% coverage.

    metrics -- str or List
        interval score metrics to compute.
        'all' returns all interval scores available
        List options: ["coverage", "acd", "winkler"]

    Returns:
    -------
    dict,
        interval score metrics within a dictionary

    Example:
    ---------
    >>> HOLDOUT = 7
    >>> PERIOD = 7

    >>> attends = load_emergency_dept()

    >>> # train-test split
    >>> train, test = attends[:-HOLDOUT], attends[-HOLDOUT:]

    >>> model = SNaive(PERIOD)

    # returns 80 and 90% prediction intervals by default.
    >>> preds, intervals = model.fit_predict(train, HOLDOUT, return_predict_int=True)

    >>> intervals_80 = intervals[0]

    >>> scores = interval_scores(test, intervals_80, alpha=0.2)
    >>> print(scores)
    """
    y_true, intervals = _validate_interval_inputs(y_true, intervals, alpha)

    if metrics == "all":
        metrics = ["coverage", "acd", "winkler"]

    funcs = _interval_evaluation_functions()
    scores = {}
    for metric in metrics:
        if metric == "coverage":
            scores[metric] = funcs[metric](y_true, intervals)
        else:
            scores[metric] = funcs[metric](y_true, intervals, alpha)

    return scores


def _interval_evaluation_functions() -> Dict:
    """
    Return all interval score functions in a dict

    Returns:
    --------
        dict
    """
    funcs = {}
    funcs["coverage"] = coverage
    funcs["acd"] = absolute_coverage_difference
    funcs["winkler"] = winkler_score
    return funcs




if __name__ == "__main__":
    y_true = [45, 60, 23, 45]
    y_preds = [50, 50, 50, 50]

    metrics = forecast_errors(y_true, y_preds)
    print(metrics)

    metrics = forecast_errors(y_true, y_preds, metrics=["mape", "smape"])
    print(metrics)
