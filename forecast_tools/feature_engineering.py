'''
forecast_tools.feature_engineering

Utilities to support the creation of features for forecasting

'''

import numpy as np

def sliding_window(train, window_size=2, horizon=1):
    '''
    Time series sliding window 

    Transforms a univariate time series into a 
    supervised learning problem with features and
    target vectors. Features are lagged observations
    in the series with lag_max = window_size.  The 
    target vector is of size horizon.
    
    Parameters:
    -----------

    train: array-like
        training data for time series method
    
    window_size: int, optional (default=2)
        lookback - the maximum lag to include in the
        features.
        
    horizon: int, optional (default=1)
        number of observations ahead to predict
            
    Returns:
    -------
        array-like, array-like
    
        preprocessed X, preprocessed Y
    '''
    tabular_X = []
    tabular_y = []
    
    for i in range(0, len(train) - window_size - horizon):
        X_train = train[i:window_size+i]
        y_train = train[i+window_size+horizon-1]
        tabular_X.append(X_train)
        tabular_y.append(y_train)
       
    return np.asarray(tabular_X), np.asarray(tabular_y).reshape(-1, 1)