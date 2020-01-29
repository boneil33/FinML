# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:35:30 2020

@author: Brendan
"""

import pandas as pd
import numpy as np
import datetime as dt
from MLFin.Preprocessing.labeling import resample_close, get_t1, get_events, get_lookback_vol
from MLFin.Research.fx_utils import fx_data_import
from MLFin.Research.FXTesting import pca_distance_loop, get_nonusd_closes
from MLFin.Preprocessing.sampling import _get_num_concurrent_events


def lookback_zscore(close, lookback, vol_lookback):
    """
    Generate lookback-period return zscores
    
    :param close: (pd.DataFrame) close prices
    :param lookback: (int) periods to look back
    :return: (pd.DataFrame) rolling lookback zscores
    """
    rets = close.apply(np.log).diff(lookback).dropna()
    rets_centered = rets.subtract(rets.rolling(vol_lookback).mean())
    vols = rets.rolling(vol_lookback).std()
    
    std_rets = rets_centered.divide(vols)
    return std_rets
    

def zscore_signal(zscores, threshold, signal_type='Reversion'):
    """
    Generate signals from zscore method
    :param zscores: (pd.DataFrame) return zscores from lookback_zscore function
    :param threshold: (int) zscore threshold at which to accept signal
    :param signal_type: (string) 'Reversion' or 'Momentum'
    :return: (pd.DataFrame) signals +1=buy, -1=sell; per security, per close index
    """
    over_thresh = zscores[abs(zscores) > threshold].fillna(0)
    signals = over_thresh.apply(np.sign)
    if signal_type=='Reversion':
        signals*=-1.
    return signals


def zscore_sizing(signals, close, vertbar, lookback, vol_lookback):
    """
    Generate events dataframe with sizes
    
    :param signals: (pd.Series) single security buy/sell signals
    :param close: (pd.Series) single security close prices
    :param vertbar: (int) time out after vertbar indices pass
    :return: (pd.DataFrame) 'events' dataframe with t1, side, size, trgt
    """
    # get events dataframe
    filtered_signals = signals[abs(signals)>0]
    t1 = get_t1(filtered_signals.index, close, vertbar)
    trgt = get_lookback_vol(close, lookback, volwindow=vol_lookback,ewma=False)
    events0 = t1.rename('t1').to_frame().merge(filtered_signals.rename('side'),
                        left_index=True, right_index=True)
    events, df0 = get_events(events0, close, trgt, pt_sl=(1,4))
    
    # generate sizing based on inverse concurrency
    concurrent = _get_num_concurrent_events(close.index, events['t1'])
    sizes = concurrent.loc[events.index].rename('size')
    events = events.merge(1./sizes, left_index=True, right_index=True)
    return events


def generate_pnl(events, close_tr):
    """
    Generate pnl series from events. Assumes close_tr and close used to 
        generate events df have same index
    
    Alternatively I should use total return index throughout this module
    
    :param events: 'events' dataframe with t1, side, size, trgt
    :param close_tr: (pd.Series) single security total return index
    :return: (pd.Series) strategy pnl series
    """
    events['ret'] = close_tr[events['t1']]/close_tr[events.index]-1.
    events['ret']*= events['side']*events['size']/events['trgt']
    
    return events
    
    
if __name__=='__main__':
    
    pair='AUDNZD'
    
    close = fx_data_import()
    close_w = resample_close(close, 'W-WED')
    
    
    group_matrix = pca_distance_loop(close, 100, 4, 0.2, components_to_use=[1,2,3])
    pair_closes = get_nonusd_closes(close, group_matrix.columns)
    zscores = lookback_zscore(pair_closes, 30, 200)
    signals = zscore_signal(zscores, 2, 'Reversion')
    group_signals = signals.multiply(group_matrix.loc[signals.index[0]:])
    events0 = zscore_sizing(group_signals[pair], pair_closes[pair], 100, 30, 100)
    events = generate_pnl(events0, pair_closes[pair])
    