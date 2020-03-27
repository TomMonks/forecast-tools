'''
Metrics to measure forecast error 
These are measures currently not found in sklearn or statsmodels
'''
import numpy as np

def mean_error(y_true, y_pred):
    '''
    Mean Error (ME)

    Parameters:
    --------
    y_true -- array-like
        actual observations from time series
    y_pred -- arraylike
        the predictions to evaluate

    Returns:
    -------
    float, 
        scalar value representing the ME
    '''
    return np.mean(y_true - y_pred)

def mean_absolute_percentage_error(y_true, y_pred): 
    '''
    Mean Absolute Percentage Error (MAPE)

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
    return np.mean(np.square((y_true - y_pred)))


def root_mean_squared_error(y_true, y_pred):
    '''
    Root Mean Squared Error (RMSE)

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
    return np.sqrt(mean_squared_error(y_true, y_pred))


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    '''
    Symmetric Mean Absolute Percentage Error (sMAPE)

    A proposed replacement for MAPE.  (But still not symmetric)

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
    numerator = 2 * np.abs(y_true - y_pred)
    denominator = np.abs(y_pred) + np.abs(y_true)
    return 100 * (numerator / denominator)



def coverage(y_true, pred_intervals):
    '''
    Calculates the proportion of the true 
    values are that are covered by the lower
    and upper bounds of the prediction intervals

    Parameters:
    -------
    y_true -- arraylike, actual observations
    pred_intervals -- np.array, matrix (hx2)
    '''
    y_true = np.asarray(y_true)
    lower = np.asarray(pred_intervals.T[0])
    upper = np.asarray(pred_intervals.T[1])
    
    cover = len(np.where((y_true > lower) & (y_true < upper))[0])
    return cover / len(y_true)