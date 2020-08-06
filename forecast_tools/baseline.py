'''
Tools for simple baseline/benchmark forecasts

These methods might serve as the forecast themselves, but are more likely
to be used as a baseline to evaluate if more complex models offer a sufficient
increase in accuracy to justify their use.

Naive1:
    Carry last value forward across forecast horizon (random walk)

SNaive:
    Carry forward value from last seasonal period

Average: np.sqrt(((h - 1) / self._period).astype(np.int)+1)
    Carry forward average of observations

Drift:
    Carry forward last time period, but allow for upwards/downwards drift.

EnsembleNaive:
    An unweighted average of all of the Naive forecasting methods.
'''

import numpy as np
import pandas as pd
from scipy.stats import norm, t
from abc import ABC, abstractmethod

# Boolean, unsigned integer, signed integer, float, complex.
_NUMERIC_KINDS = set('buifc')


def is_numeric(array):
    """Determine whether the argument has a numeric datatype, when
    converted to a NumPy array.

    Booleans, unsigned integers, signed integers, floats and complex
    numbers are the kinds of numeric datatype.

    source:
    https://codereview.stackexchange.com/questions/128032/check-if-a-numpy-array-contains-numerical-data

    Parameters
    ----------
    array : array-like
        The array to check.

    Returns
    -------
    is_numeric : `bool`
        True if the array has a numeric datatype, False if not.

    """
    return np.asarray(array).dtype.kind in _NUMERIC_KINDS


class Forecast(ABC):
    '''
    Abstract base class for all baseline forecast
    methods
    '''

    def __init__(self):
        self._fitted = None
        self._t = None

    def _get_fitted(self):
        return self._fitted['pred']

    def _get_resid(self):
        return self._fitted['resid']

    @abstractmethod
    def fit(self, train):
        pass

    def fit_predict(self, train, horizon, return_predict_int=False,
                    alpha=None):
        '''
        Convenience method.  Fit model and predict with one call.

        Parameters:
        ---------

        train: array-like,
            vector, series, or dataframe of the time series used for training.
            Values should be floats and not contain any np.nan or np.inf

        horizon: int,
            forecast horizon.

        return_predict_int: bool, optional (default=False)
            If True function will return a Tuple
            0: point forecasts (mean)
            1: matrix of intervals.

        alpha: None, or list of floats, optional (default=None)
            List of floats between 0 and 1. If return_predict_int == True this
            specifies the 100(1-alpha) prediction intervals to return.

        Returns:
        ------
        np.array, vector of predictions. length=horizon

        '''
        self.fit(train)
        return self.predict(horizon, return_predict_int=return_predict_int,
                            alpha=alpha)

    def validate_training_data(self, train, min_length=1):
        '''
        Checks the validity of training data for forecasting
        and raises exceptions if required.

        1. check is instance of pd.Series, pd.DataFrame or np.ndarray
        2. check len is > min_length

        Parameters:
        ---------
        min_length: int optional (default=0)
            minimum length of the time series.

        '''

        if not isinstance(train, (pd.Series, pd.DataFrame, np.ndarray)):
            raise TypeError(
                'Training data must be pd.Series, pd.DataFrame or np.ndarray')
        elif len(train) < min_length:
            raise ValueError('Training data is empty')
        elif not is_numeric(train):
            raise TypeError('Training data must be numeric')

        elif np.isnan(np.asarray(train)).any():
            raise TypeError(
                'Training data contains at least one NaN. '
                + 'Data myst all be floats')
        elif np.isinf(np.asarray(train)).any():
            raise TypeError(
                'Training data contains at least one Infinite '
                + 'value (np.Inf). Data myst all be floats')

    @abstractmethod
    def predict(self, horizon, return_predict_int=False, alpha=None):
        pass

    def _prediction_interval(self, horizon, alpha=None):
        '''
        Prediction intervals for naive forecast 1 (NF1)

        lower = pred - z * std_h
        upper = pred + z * std_h

        where

        std_h = resid_std * sqrt(h)

        resid_std = standard deviation of in-sample residuals

        h = horizon

        See and credit: https://otexts.com/fpp2/prediction-intervals.html

        Pre-requisit: Must have called fit()

        Parameters:
        ---------
        horizon - int,
                    forecast horizon

        levels - list,
            list of floats representing prediction limits
            e.g. [0.80, 0.90, 0.95] will calculate three sets ofprediction
            intervals giving limits for which will include the actual future
            value with probability 80, 90 and 95 percent,
            respectively (default = [0.8, 0.95]).

        Returns:
        --------
        list
            np.array matricies that contain the lower and upper prediction
            limits for each prediction interval specified.

        '''

        if alpha is None:
            alpha = [0.20, 0.05]

        zs = [self.interval_multiplier(1-a, self._t - 1) for a in alpha]

        pis = []

        std_h = self._std_h(horizon)

        for z in zs:
            hw = z * std_h
            pis.append(np.array([self.predict(horizon) - hw,
                                 self.predict(horizon) + hw]).T)

        return pis

    def interval_multiplier(self, level, dof):
        '''
        inverse of normal distribution
        can be overridden if needed.
        '''
        x = norm.ppf((1 - level) / 2)
        return np.abs(x)

    @abstractmethod
    def _std_h(self, horizon):
        '''
        Calculate the standard error of the residuals over
        a forecast horizon.  This is method specific.
        '''
        pass

    # breaks PEP8 to align with statsmodels naming
    fittedvalues = property(_get_fitted)
    resid = property(_get_resid)


class Naive1(Forecast):
    '''
    Naive forecast 1 or NF1: Carry the last value foreward across a
    forecast horizon

    For details and theory see [1]

    Attributes
    ----------
    fittedvalues: pd.Series
        In-sample predictions of training data
    resid: pd.Series
        In-sample residuals

    Methods
    -------
    fit(train)
        fit the model to training data
    predict(horizon, return_predict_int=False, alpha=None)
        Predict h-steps ahead
    fit_predict(train, horizons, return_predict_int=False, alpha=None)
        convenience method.  combine fit() and predict()

    See Also
    --------
    forecast_tools.baseline.SNaive
    forecast_tools.baseline.Drift
    forecast_tools.baseline.Average
    forecast_tools.baseline.EnsembleNaive

    References:
    ----------
    [1]. https://otexts.com/fpp2/simple-methods.html

    Examples:
    --------

    Basic fitting and prediction

    >>> y_train = np.arange(10)
    >>> model = Naive1()
    >>> model.fit(y_train)
    >>> model.predict(horizon=7)
    array([9., 9., 9., 9., 9., 9., 9.]

    fit_predict() convenience method

    >>> y_train = np.arange(10)
    >>> model = Naive1()
    >>> model.fit_predict(y_train, horizon=7)
    array([9., 9., 9., 9., 9., 9., 9.]

    80 and 95% prediction intervals

    >>> y_train = np.arange(10)
    >>> model = Naive1()
    >>> model.fit(y_train)
    >>> y_pred, y_intervals = model.predict(horizon=2,
                                            return_pred_interval=True,
                                            alpha=[0.1, 0.05])
    >>> y_pred
    array([9., 9.]
    >>> y_intervals[0]
    array([[ 7.71844843, 10.28155157],
           [ 7.1876124 , 10.8123876 ]])
    >>> y_intervals[1]
    array([[ 7.35514637, 10.64485363],
           [ 6.67382569, 11.32617431]])


    Fitted values (one step in-sample predictions)
    .fittedvalue returns a pandas.Series called pred

    >>> y_train = np.arange(5)
    >>> model = Naive1()
    >>> model.fit(y_train)
    >>> model.fittedvalues
    0    NaN
    1    0.0
    2    1.0
    3    2.0
    4    3.0
    Name: pred, dtype: float64

    '''
    def __init__(self):
        '''
        Constructor method

        Parameters:
        -------
        level - list,
            confidence levels for prediction intervals (e.g. [90, 95])
        '''
        self._fitted = None

    def __repr__(self):
        '''
        String representation of object
        '''
        return f'Naive1()'

    def __str__(self):
        '''
        Print/str representation of object
        '''
        return f'Naive1()'

    def fit(self, train):
        '''
        Train the naive model

        Parameters:
        --------
        train - array-like,
            vector, series, or dataframe of the time series used for training.
            Values should be floats and not contain any np.nan or np.inf
        '''

        self.validate_training_data(train)

        _train = np.asarray(train)
        self._pred = _train[-1]
        self._fitted = pd.DataFrame(_train)

        if isinstance(train, (pd.DataFrame, pd.Series)):
            self._fitted.index = train.index

        self._t = len(_train)
        self._fitted.columns = ['actual']
        self._fitted['pred'] = self._fitted['actual'].shift(periods=1)
        self._fitted['resid'] = self._fitted['actual'] - self._fitted['pred']
        self._resid_std = np.sqrt(np.nanmean(np.square(self._fitted['resid'])))

    def predict(self, horizon, return_predict_int=False, alpha=None):
        '''
        Forecast and optionally produce 100(1-alpha) prediction intervals.

        Prediction intervals for naive forecast 1 (NF1)

        lower = pred - z * std_h
        upper = pred + z * std_h

        where

        std_h = resid_std * sqrt(h)

        resid_std = standard deviation of in-sample residuals

        h = horizon

        See and credit: https://otexts.com/fpp2/prediction-intervals.html

        Pre-requisit: Must have called fit()

        Parameters:
        --------
        horizon - int,
                    forecast horizon.

        return_predict_int: bool, optional
                    if True calculate 100(1-alpha) prediction
                intervals for the forecast. (default=False)

        alpha: list of floats, optional (default=None)
            controls set of prediction intervals returned and the width of
            each.

            Intervals are 100(1-alpha) in width. e.g. [0.2, 0.1]
            would return the 80% and 90% prediction intervals of the forecast
            distribution.  default=None.  When return_predict_int = True the
            default behaviour is to return 80 and 90% intervals.

        Returns:
        -------

        if return_predict_int = False

        np.array, vector of predictions. length=horizon

        if return_predict_int = True then returns a tuple.

        0. np.array, vector of predictions. length=horizon
        1. list of numpy.array[lower_pi, upper_pi].
            One for each prediction interval.

        '''
        if self._fitted is None:
            raise UnboundLocalError('Must call fit() prior to predict()')

        if alpha is None:
            alpha = [0.2, 0.1]

        preds = np.full(shape=horizon, fill_value=self._pred, dtype=float)

        if return_predict_int:
            return preds, self._prediction_interval(horizon, alpha)
        else:
            return preds

    def _std_h(self, horizon):
        '''
        Calculate the sample standard deviation.
        '''
        indexes = np.sqrt(np.arange(1, horizon+1))

        std = np.full(shape=horizon,
                      fill_value=self._resid_std,
                      dtype=np.float)

        std_h = std * indexes
        return std_h


class SNaive(Forecast):
    '''
    Seasonal Naive Forecast SNF

    Each forecast to be equal to the last observed value from the
    same season of the year (e.g., the same month of the previous year).

    SNF is useful for highly seasonal data. See [1]

    Attributes
    ----------
    fittedvalues: pd.Series
        In-sample predictions of training data
    resid: pd.Series
        In-sample residuals

    Methods
    -------
    fit(train)
        fit the model to training data
    predict(horizon, return_predict_int=False, alpha=None)
        Predict h-steps ahead
    fit_predict(train, horizons, return_predict_int=False, alpha=None)
        convenience method.  combine fit() and predict()

    See Also
    --------
    forecast_tools.baseline.Naive1
    forecast_tools.baseline.Drift
    forecast_tools.baseline.Average
    forecast_tools.baseline.EnsembleNaive

    References:
    -----------
    [1]. https://otexts.com/fpp2/simple-methods.html

    '''

    def __init__(self, period):
        '''
        Parameters:
        --------
        period - int, the seasonal period of the daya
                 e.g. weekly = 7, monthly = 12, daily = 24
        '''
        self._period = period
        self._fitted = None

    def __repr__(self):
        '''
        String representation of object
        '''
        return f'SNaive1(period={self._period})'

    def __str__(self):
        '''
        Print/str representation of object
        '''
        return f'SNaive1(period={self._period})'

    def fit(self, train):
        '''
        Seasonal naive forecast - train the model

        Parameters:
        --------
        train: array-like.
            vector, pd.DataFrame or pd.Series containing the time series used
            for training. Values should be floats and not contain any np.nan
            or np.inf
        '''

        self.validate_training_data(train, min_length=self._period)

        # could refactor this to be more like Naive1's simpler implementation.
        if isinstance(train, (pd.Series)):
            self._f = np.asarray(train)[-self._period:]
            _train = train.to_numpy()
            self._fitted = pd.DataFrame(_train, index=train.index)
        elif isinstance(train, (pd.DataFrame)):
            self._f = train.to_numpy().T[0][-self._period:]
            _train = train.copy()[train.columns[0]].to_numpy()
            self._fitted = pd.DataFrame(_train, index=train.index)
        else:
            self._f = train[-self._period:]
            _train = train.copy()
            self._fitted = pd.DataFrame(_train)

        self._t = len(_train)
        self._fitted.columns = ['actual']
        self._fitted['pred'] = self._fitted['actual'].shift(self._period)
        self._fitted['resid'] = self._fitted['actual'] - self._fitted['pred']

        self._resid_std = np.sqrt(np.nanmean(np.square(self._fitted['resid'])))

    def predict(self, horizon, return_predict_int=False, alpha=None):
        '''
        Predict time series over a horizon

        Parameters:
        --------
        horizon - int,
            forecast horizon.

        return_predict_int: bool, optional
                    if True calculate 100(1-alpha) prediction
                intervals for the forecast. (default=False)

        alpha: list of floats, optional (default=None)
            controls set of prediction intervals returned and the width of
            each.

            Intervals are 100(1-alpha) in width. e.g. [0.2, 0.1]
            would return the 80% and 90% prediction intervals of the forecast
            distribution.  default=None.  When return_predict_int = True the
            default behaviour is to return 80 and 90% intervals.

        Returns:
        --------

        if return_predict_int = False

        np.array, vector of predictions. length=horizon

        if return_predict_int = True then returns a tuple.

        0. np.array, vector of predictions. length=horizon
        1. list of numpy.array[lower_pi, upper_pi].
            One for each prediction interval.
        '''

        if self._fitted is None:
            raise UnboundLocalError('Must call fit() prior to predict()')

        if alpha is None:
            alpha = [0.2, 0.1]

        preds = np.array([], dtype=float)

        for _ in range(0, int(horizon/self._period)):
            preds = np.concatenate([preds, self._f.copy()], axis=0)

        preds = np.concatenate([preds,
                               self._f.copy()[:horizon % self._period]],
                               axis=0)

        if return_predict_int:
            return preds, self._prediction_interval(horizon, alpha)
        else:
            return preds

    def _std_h(self, horizon):
        h = np.arange(1, horizon+1)
        # need to query if should be +1 or not.
        return self._resid_std * \
            np.sqrt(((h - 1) / self._period).astype(np.int)+1)


class Average(Forecast):
    '''
    Average forecast.  Forecast is set to the average
    of the historical data.

    See for discussion of the average as a forecat measure [1]

    Attributes
    ----------
    fittedvalues: pd.Series
        In-sample predictions of training data
    resid: pd.Series
        In-sample residuals

    Methods
    -------
    fit(train)
        fit the model to training data
    predict(horizon, return_predict_int=False, alpha=None)
        Predict h-steps ahead
    fit_predict(train, horizons, return_predict_int=False, alpha=None)
        convenience method.  combine fit() and predict()

    See Also
    --------
    forecast_tools.baseline.Naive1
    forecast_tools.baseline.SNaive
    forecast_tools.baseline.Drift
    forecast_tools.baseline.EnsembleNaive

    References:
    -----------
    [1.] Makridakis, Wheelwright and Hyndman. Forecasting (1998)
    '''

    def __init__(self):
        self._pred = None
        self._fitted = None

    def __repr__(self):
        '''
        String representation of object
        '''
        return f'Average()'

    def __str__(self):
        '''
        Print/str representation of object
        '''
        return f'Average()'

    def _get_fitted(self):
        return self._fitted['pred']

    def _get_resid(self):
        return self._fitted['resid']

    def fit(self, train):
        '''
        Train the model

        Parameters:
        --------
        train:  arraylike
                vector, pd.series, pd.DataFrame,
                Time series used for training.  Values should be floats
                and not contain any np.nan or np.inf
        '''

        self.validate_training_data(train)

        if isinstance(train, (pd.DataFrame)):
            _train = train.copy()[train.columns[0]].to_numpy()
            self._fitted = pd.DataFrame(_train, index=train.index)
        elif isinstance(train, (pd.Series)):
            _train = train.to_numpy()
            self._fitted = pd.DataFrame(_train, index=train.index)
        else:
            _train = train.copy()
            self._fitted = pd.DataFrame(train)

        self._fitted.columns = ['actual']

        self._t = len(_train)
        self._pred = _train.mean()
        # ddof set to get sample mean
        self._resid_std = (_train - self._pred).std(ddof=1)
        self._fitted['pred'] = self._pred
        self._fitted['resid'] = self._fitted['actual'] - self._fitted['pred']

    def predict(self, horizon, return_predict_int=False, alpha=None):
        '''
        Predict time series over a horizon

        Parameters:
        --------
        horizon - int, forecast horizon.

        return_predict_int: bool, optional
                    if True calculate 100(1-alpha) prediction
                intervals for the forecast. (default=False)

        alpha: list of floats, optional (default=None)
            controls set of prediction intervals returned and the width of
            each.

            Intervals are 100(1-alpha) in width. e.g. [0.2, 0.1]
            would return the 80% and 90% prediction intervals of the forecast
            distribution.  default=None.  When return_predict_int = True the
            default behaviour is to return 80 and 90% intervals.


        Returns:
        --------

        if return_predict_int = False

        np.array, vector of predictions. length=horizon

        if return_predict_int = True then returns a tuple.

        0. np.array, vector of predictions. length=horizon
        1. list of numpy.array[lower_pi, upper_pi].
            One for each prediction interval.
        '''

        if self._fitted is None:
            raise UnboundLocalError('Must call fit() prior to predict()')

        if alpha is None:
            alpha = [0.2, 0.1]

        preds = np.full(shape=horizon, fill_value=self._pred, dtype=float)

        if return_predict_int:
            return preds, self._prediction_interval(horizon, alpha)
        else:
            return preds

    def interval_multiplier(self, level, dof):
        '''
        inverse of student t distribution
        '''
        x = t.ppf((1 - level) / 2, dof)
        return np.abs(x)

    def _std_h(self, horizon):
        std = self._resid_std * np.sqrt(1 + (1/self._t))
        return np.full(shape=horizon, fill_value=std, dtype=np.float)


class Drift(Forecast):
    '''
    Naive1 forecast with drift

    Carry the last value foreward across a forecast horizon but
    allow for upwards of downwards drift defined in [1]

    Drift = average change in the historical data.

    Note.  The current implementation has a standard error of the forecast
    that is the same as for the naive1 se.  This could be adjusted for drift.
    The following link suggests this is minor and benchmark with R is v.similar
    [2]

    Attributes
    ----------
    fittedvalues: pd.Series
        In-sample predictions of training data
    resid: pd.Series
        In-sample residuals

    Methods
    -------
    fit(train)
        fit the model to training data
    predict(horizon, return_predict_int=False, alpha=None)
        Predict h-steps ahead
    fit_predict(train, horizons, return_predict_int=False, alpha=None)
        convenience method.  combine fit() and predict()

    See Also
    --------
    forecast_tools.baseline.Naive1
    forecast_tools.baseline.SNaive
    forecast_tools.baseline.Average
    forecast_tools.baseline.EnsembleNaive

    References:
    -----------
    [1]. https://otexts.com/fpp2/simple-methods.html
    [2]. https://www.coursehero.com/file/p12k3ln/For-the-random-walk-with-drift-model-the-1-step-ahead-forecast-standard-error/

    '''

    def __init__(self):
        self._fitted = None

    def __repr__(self):
        '''
        String representation of object
        '''
        return f'Drift()'

    def __str__(self):
        '''
        Print/str representation of object
        '''
        return f'Drift()'

    def _get_fitted_gradient(self):
        return self._fitted['gradient_fit']

    def fit(self, train):
        '''
        Train the naive with drift model

        Parameters:
        --------
        train:  arraylike
                vector, pd.series, pd.DataFrame,
                Time series used for training.  Values should be floats
                and not contain any np.nan or np.inf

        '''

        self.validate_training_data(train)

        # if dataframe convert to series for compatability with
        # proc (for convenience of passing the dataframe rather than a series)
        if isinstance(train, (pd.DataFrame)):
            _train = train.copy()[train.columns[0]].to_numpy()
            self._fitted = pd.DataFrame(_train, index=train.index)
        elif isinstance(train, (pd.Series)):
            _train = train.to_numpy()
            self._fitted = pd.DataFrame(_train, index=train.index)
        else:
            _train = train.copy()
            self._fitted = pd.DataFrame(train)

        self._last_value = _train[-1:][0]
        self._t = _train.shape[0]
        self._gradient = ((self._last_value - _train[0]) / (self._t - 1))
        self._fitted.columns = ['actual']

        # could show fitted as line from first to last point.
        # unclear if should use this or naive1 method.
        self._fitted['gradient_fit'] = _train[0] \
            + np.arange(1, self._t+1, dtype=float) * self._gradient

        # 1 step carry forward naive1 fitted values.
        self._fitted['pred'] = self._fitted['actual'].shift(periods=1)
        self._fitted['resid'] = self._fitted['actual'] - self._fitted['pred']

        # standard error is not adjusted for drift - how much of an issue?
        self._resid_std = np.sqrt(np.nanmean(np.square(self._fitted['resid'])))

    def predict(self, horizon, return_predict_int=False, alpha=None):
        '''
        Parameters:
        --------
        horizon - int, forecast horizon.

        return_predict_int: bool, optional
                    if True calculate 100(1-alpha) prediction
                intervals for the forecast. (default=False)

        alpha: list of floats, optional (default=None)
            controls set of prediction intervals returned and the width of
            each.

            Intervals are 100(1-alpha) in width. e.g. [0.2, 0.1]
            would return the 80% and 90% prediction intervals of the forecast
            distribution.  default=None.  When return_predict_int = True the
            default behaviour is to return 80 and 90% intervals.


        Returns:
        --------

        if return_predict_int = False

        np.array, vector of predictions. length=horizon

        if return_predict_int = True then returns a tuple.

        0. np.array, vector of predictions. length=horizon
        1. list of numpy.array[lower_pi, upper_pi].
            One for each prediction interval.
        '''

        if self._fitted is None:
            raise UnboundLocalError('Must call fit() prior to predict()')

        if alpha is None:
            alpha = [0.2, 0.1]

        preds = np.arange(1, horizon+1, dtype=float) * self._gradient
        preds += self._last_value

        if return_predict_int:
            return preds, self._prediction_interval(horizon, alpha)
        else:
            return preds

    def _std_h(self, horizon):

        h = np.arange(1, horizon+1)
        return self._resid_std * np.sqrt(h * (1 + (h / self._t)))

    fitted_gradient = property(_get_fitted_gradient)


class EnsembleNaive(Forecast):
    '''
    An ensemble of all naive forecast methods.

    Attributes
    ----------
    fittedvalues: pd.Series
        In-sample predictions of training data
    resid: pd.Series
        In-sample residuals

    Methods
    -------
    fit(train)
        fit the model to training data
    predict(horizon, return_predict_int=False, alpha=None)
        Predict h-steps ahead
    fit_predict(train, horizons, return_predict_int=False, alpha=None)
        convenience method.  combine fit() and predict()

    See Also
    --------
    forecast_tools.baseline.Naive1
    forecast_tools.baseline.SNaive
    forecast_tools.baseline.Average
    forecast_tools.baseline.Drift
    '''

    def __init__(self, seasonal_period):
        self._estimators = {'NF1': Naive1(),
                            'SNaive': SNaive(period=seasonal_period),
                            'Average': Average(),
                            'Drift': Drift()
                            }

    def __repr__(self):
        '''
        String representation of object
        '''
        p = self._estimators['SNaive']._period
        return f'EnsembleNaive(seasonal_period={p})'

    def __str__(self):
        '''
        Print/str representation of object
        '''
        p = self._estimators['SNaive']._period
        return f'EnsembleNaive(seasonal_period={p})'

    def fit(self, train):
        '''
        Parameters:
        --------

        train:  arraylike
                vector, pd.series, pd.DataFrame,
                Time series used for training.  Values should be floats
                and not contain any np.nan or np.inf
        '''

        for _, estimator in self._estimators.items():
            estimator.fit(train)

    def predict(self, horizon, return_predict_int=False, alpha=None):
        preds = []
        for _, estimator in self._estimators.items():
            model_preds = estimator.predict(horizon)
            preds.append(model_preds)

        return np.array(preds).mean(axis=0)

    def _std_h(self, horizon):
        '''
        Calculate the standard error of the residuals over
        a forecast horizon.  This is method specific.
        '''
        pass


def baseline_estimators(seasonal_period):
    '''
    Generate a collection of baseline forecast objects

    Parameters:
    --------
    seasonal_period - int,
        order of seasonal periods in the data (e.g daily = 7)

    average_lookback - int,
        number of lagged periods that average baseline includes

    Returns:
    --------
    dict,
        forecast objects
    '''

    estimators = {'NF1': Naive1(),
                  'SNaive': SNaive(period=seasonal_period),
                  'Average': Average(),
                  'Drift': Drift(),
                  'Ensemble': EnsembleNaive(seasonal_period=seasonal_period)
                  }

    return estimators


def boot_prediction_intervals(preds, resid, horizon, levels=None, boots=1000):
    '''
    Constructs bootstrap prediction intervals for forecasting.

    This procedure makes no assumptions about the distribution of errors
    e.g if they are normally distributed, but does assumes that no significant
    autocorrelation exists in residuals.

    Parameters:
    -----------

    preds: array-like,
        predictions over forecast horizon

    resid: array-like,
        in-sample prediction residuals

    horizon: int,
        forecast horizon (e.g. 12 months or 7 days)

    levels: list of floats,
        prediction interval precisions (default=[0.80, 0.95])

    boots: int,
        number of bootstrap datasets to construct (default = 1000)

    Returns:
    ---------

    list of numpy arrays.  Each numpy array contains two columns of the upper
    and lower prediction limits across the forecast horizon.
    '''

    if levels is None:
        levels = [0.80, 0.95]

    resid = _drop_na_from_series(resid)

    sample = np.random.choice(resid, size=(boots, horizon))
    sample = np.cumsum(sample, axis=1)

    data = preds + sample

    pis = []
    for level in levels:

        alpha = (1 - level) / 2
        q_upper = level + alpha
        q_lower = (1 - level) - alpha

        upper = np.percentile(data, q_upper*100, interpolation='higher',
                              axis=0)
        lower = np.percentile(data, q_lower*100, interpolation='higher',
                              axis=0)

        pis.append(np.array([lower, upper]).T)

    return pis


def _drop_na_from_series(data):
    '''
    Drops all NaN from numpy array or pandas series.

    Parameters:
    -------
    data, array-like,
    np.ndarray or pd.Series.

    Returns:
    -------
    np.ndarray removing NaN.

    '''
    if isinstance(data, pd.Series):
        return data.dropna().to_numpy()
    else:
        return data[~np.isnan(data)]
