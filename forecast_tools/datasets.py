'''
Contains functions for loading built-in datasets
'''

import pandas as pd
import os

PATH_ED = 'data/ed_ts.csv'


def load_emergency_dept():
    '''
    344 simulated daily level attendenances at a single
    emergency department between 22/01/2017
    and 31/02/2017 (dates are UK dd/mm/yyyy format)

    The returned data frame has a DateTimeIndex with
    freq 'D' and value column 'arrivals' shape = (344, 1)

    Returns
    -------
        pandas.DataFrame
    '''
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(dir_path, PATH_ED)
    df = pd.read_csv(path, index_col='date', parse_dates=True,
                     dayfirst=True)
    df.index.freq = 'D'
    return df
