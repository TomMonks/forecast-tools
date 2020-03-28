'''
Contains functions for loading built-in datasets
'''

import pandas as pd
import os

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
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(dir_path, PATH_ED)
    df = pd.read_csv(path, index_col='date', parse_dates=True, 
                     dayfirst=True)
    df.index.freq = 'D'
    return df

if __name__ == '__main__':
    df = load_emergency_dept()
    print(df.shape)
    print(df.index.freq)
    print(df.head())