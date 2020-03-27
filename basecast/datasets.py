'''
Contains functions for loading built-in datasets
'''

import pandas as pd

PATH_ED = 'data/ed_ts.csv'

def load_emergency_dept():
    '''
    Daily level attendenances at a single
    emergency department between 22/01/2014 
    and 31/02/2014 (dates are UK dd/mm/yyyy format)

    Returns
    -------
        pandas.DataFrame
    '''
    df = pd.load_csv(PATH_ED, index_col='date', parse_dates=True)
    df.index.freq = 'MS'
    return df