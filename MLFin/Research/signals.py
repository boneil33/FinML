# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:35:30 2020

@author: Brendan
"""

import pandas as pd
import numpy as np
import datetime as dt
from Preprocessing.labeling import resample_close, get_t1, get_events, get_lookback_vol
from Research.fx_utils import fx_data_import, bbg_data_import
from Research.FXTesting import pca_distance_loop, get_nonusd_pair_data, get_nonusd_pairs
from Preprocessing.sampling import get_num_conc_events_side, get_max_concurrency
from Preprocessing.etf_trick import ETFTrick


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


def zscore_sizing(signals, close, vertbar, lookback, vol_lookback, pt_sl=(1,1)):
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
    events, df0 = get_events(events0, close, trgt, pt_sl=pt_sl)
    
    # generate sizing based on inverse concurrency
    if len(events)==0:
        return None
    concurrent = get_num_conc_events_side(close.index, events)
    max_conc = get_max_concurrency(events, concurrent)
    max_conc = max_conc.loc[events.index].rename('max_conc')
    events = events.merge(max_conc, left_index=True, right_index=True)
    events = events.merge(concurrent, left_index=True, right_index=True)
    events['size'] = 1./events.loc[:, 'max_conc']
    events['size']*= events.apply(lambda x: x['long'] if x['side']>0 else (
                                      x['short'] if x['side']<0 else 0), axis=1)
    # if this is the first run, size is 0.1
    max_long = ((events['side']>0) & (events['long']==events['max_conc']))
    max_short = ((events['side']<0) & (events['short']==events['max_conc']))
    events['size'] = events['size'].mask(max_long, 0.1)
    events['size'] = events['size'].mask(max_short, 0.1)
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
    
    events['ret'] = close_tr.loc[events['t1']].values/close_tr.loc[events.index].values-1.
    events['ret']*= events['side']*events['size']/events['trgt']
    
    return events
    

def generate_mtm_pnl(events, close_tr, log_diff=True):
    """
    Generate daily mark to market pnl from events.
    todo: paralellize this to test yourself!
    
    :param events: 'events' dataframe with t1, side, size, trgt
    :param close_tr: (pd.Series) single security total return index
    :return: (pd.Series) mark to market pnl per index on close_tr
    """
    events = events.copy(deep=True)
    close = close_tr.copy(deep=True)
    
    events.loc[:, 't1'] = events.loc[:, 't1'].fillna(close_tr.index[-1])
    
    if log_diff:
        close = close.apply(np.log)
    close_diff = close.diff(1).fillna(0)
    pnl_df = pd.Series(0, index=close_tr.index)
    for loc, row in events.itertuples():
        pnl_df.loc[loc:t1] += close_diff.loc[loc:t1]*row.side*row.size*row.trgt
    
    #re-exponentiate
    if log_diff:
        pnl_df = pnl_df.apply(np.exp)-1.
    
    return pnl_df

    
if __name__=='__main__':
    # implement citi reversion
    # start with just AUDNZD
    pair='AUDNZD'
    c1 = 'USDAUD'
    c2 = 'USDNZD'
    close, yields = bbg_data_import(vs_dollar=True)
    #close_w = resample_close(close, 'W-FRI')
    crosses, cross_yields = get_nonusd_pair_data(close, yields, ['AUDNZD'])
    cross_yields = cross_yields.fillna(0)/100./365.
    weights = pd.DataFrame(np.ones(crosses.shape), index=crosses.index, columns=crosses.columns)
    tr = ETFTrick(crosses.shift(1), crosses, weights, cross_yields)
    tr_s = tr.get_etf_series()

#   if we wanted to drop the union of NAs
#    cross_yields = crosses.merge(yields, left_index=True, right_index=True, suffixes=('_px','_yld'))

    nonusd_pairs = get_nonusd_pairs(close.columns)
    group_matrix = pca_distance_loop(close, 100, 4, 0.2, nonusd_pairs, components_to_use=[1,2,3])
    pair_closes, yields = get_nonusd_pair_data(close, yields, nonusd_pairs)
    zscores = lookback_zscore(pair_closes, 30, 200)
    signals = zscore_signal(zscores, 2, 'Reversion')
    group_signals = signals.multiply(group_matrix.loc[signals.index[0]:])
    events0 = zscore_sizing(group_signals[pair], pair_closes[pair], 300, 30, 100)
    events = generate_pnl(events0, tr_s)
    

    
