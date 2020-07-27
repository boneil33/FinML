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


def lookback_zscore(close, lookback, vol_lookback, log_ret=True, center=True):
    """
    Generate lookback-period return zscores
    
    :param close: (pd.DataFrame) close prices
    :param lookback: (int) periods to look back
    :param vol_lookback: (int) rolling window size
    :return: (pd.DataFrame) rolling lookback zscores
    """
    if log_ret:
        close = close.apply(np.log)

    rets = close.diff(lookback).dropna()
    
    if center:
        rets = rets.subtract(rets.rolling(vol_lookback).mean())

    vols = rets.rolling(vol_lookback).std()    
    std_rets = rets.divide(vols)
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


def get_sizes(events, discrete_width, prob, pred=None, num_classes=2):
    """
        Determine bet sizes from predicted probabilities
    
    :param events: 'events' dataframe with t1, side, size, trgt
    :param discrete_width: (float) fraction for discretization of bet sizes
    :param prob: (pd.Series) predicted probability of each event
    :param pred: (pd.Series) predicted label of each event
    :param num_classes: (int) number of potential labels
    :return: (pd.Series) bet sizes for each event in [-1.0, 1.0]
    """
def zscore_sizing(signals, close, vertbar, lookback=1, vol_lookback=100, pt_sl=(1,1), 
                  trgt=None, model_resids=None, max_signals=None, even_weight=False):
    """
        Generate events dataframe with sizes. 
        Size is the trade size, each signal being a different trade.
    
    :param signals: (pd.Series) single security buy/sell signals
    :param close: (pd.Series) single security close prices
    :param vertbar: (int) time out after vertbar indices pass
    :param model_resids: (pd.Series) (optional) external model residuals to pass to get_events
    :param max_signals: (int) (optional) max number of concurrent signals to accept
    :return: (pd.DataFrame) 'events' dataframe with t1, side, size, trgt
    """
    # get events dataframe
    filtered_signals = signals[abs(signals)>0]
    t1 = get_t1(filtered_signals.index, close, vertbar)
    if trgt is None:
        trgt = get_lookback_vol(close, lookback, volwindow=vol_lookback,ewma=False)
    events0 = t1.rename('t1').to_frame().merge(filtered_signals.rename('side'),
                        left_index=True, right_index=True)
    events, df0 = get_events(events0, close, trgt, pt_sl=pt_sl,model_resids=model_resids)
    
    # generate sizing based on inverse concurrency
    if len(events)==0:
        return None
    concurrent = get_num_conc_events_side(close.index, events)
    max_conc = get_max_concurrency(events, concurrent)
    max_conc = max_conc.loc[events.index].rename('max_conc')
    events = events.merge(max_conc, left_index=True, right_index=True)
    events = events.merge(concurrent, left_index=True, right_index=True)
    events['size'] = 1.
    if not even_weight:
        events['size']/= events.loc[:, 'max_conc']
        events['size']*= events.apply(lambda x: x['long'] if x['side']>0 else (
                                          x['short'] if x['side']<0 else 0
                                          ), axis=1)
    
    # if we've never had that many concurrent trades, size is 0.1
    # handle unnamed index
    if events.index.name is not None:
        events_index_name = events.index.name
    else:
        events.index.name = 'Date'
        events_index_name = 'Date'
    events.reset_index(inplace=True)
    max_long = ((events['side']>0) & (events['long']==events['max_conc']))
    max_short = ((events['side']<0) & (events['short']==events['max_conc']))
    events.loc[:,'size'] = events.loc[:,'size'].mask(max_long, 0.1)
    events.loc[:,'size'] = events.loc[:,'size'].mask(max_short, 0.1)
    
    # if max_signals, drop consecutive trades after that number of signals
    if max_signals is not None:
        events = events[~((events['long']>max_signals) | (events['short']>max_signals))]
    
    events.set_index(events_index_name, inplace=True)
    
    return events


def generate_pnl(events, close_tr, pct_pnl=True):
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
    if 'size' not in events.columns:
        events['size'] = 1.
    if 'trgt' not in events.columns:
        events['trgt'] = 1.
    if log_diff:
        close = close.apply(np.log)
    close_diff = close.diff(1).fillna(0)
    pnl_df = pd.Series(0, index=close_tr.index)
    for row in events.itertuples():
        pnl_df.loc[row.Index:row.t1] += close_diff.loc[row.Index:row.t1]*row.side*row.size*row.trgt
    
    #re-exponentiate
    if log_diff:
        pnl_df = pnl_df.apply(np.exp)-1.
    
    return pnl_df


def generate_exposures(events, close):
    """
    Generate exposure per day
    """
    exposures = pd.Series(0, index=close.index)
    for row in events.itertuples():
        exposures.loc[row.Index:row.t1] += row.side*row.size
    
    return exposures


def generate_pnl_index(mtm_pnl):
    """
    Generate strategy total return index from mtm_pnl
    
    Effectively just treat it as carry and use ETFTrick.get_etf_series()
    :param mtm_pnl: (pd.Series) $pnl series from generate_mtm_pnl function
    """
    df1 = pd.DataFrame(1, index=mtm_pnl.index, columns=['strat'])
    
    trick = ETFTrick(df1, df1, df1, mtm_pnl.rename('strat').to_frame())
    index_pnl = trick.get_etf_series()
    return index_pnl

    
def generate_perf_summary(events, close_tr):
    """
    Function to generate CAGR, vol, sharpe, calmar, max drawdown, # trades, avg pnl per trade, hit ratio
    
    :input events: (pd.DataFrame) 'events' dataframe with t1, side, size, trgt
    :input close_tr: (pd.Series) total return series of underlying product
    :return: (pd.DataFrame) summary of pnl attributes
    """
    pnl = generate_mtm_pnl(events, close_tr, log_diff=True)
    pnl_index = generate_pnl_index(pnl)
    last_date = min(np.hstack((events['t1'].iloc[-1],pnl_index.index[-1])))
    years_live = (last_date-events.index[0]).days/365.25
    cagr = np.power(pnl_index.iloc[-1]/pnl_index.iloc[0],1/years_live)-1.
    
    log_returns = pnl_index.apply(np.log).diff(1).dropna()
    vol = log_returns.std()
    # assumes daily close data
    annualized_vol = vol*np.sqrt(252)
    sharpe = cagr/annualized_vol
    drawdown_pct = pnl_index.divide(pnl_index.expanding(0).max())-1.
    max_dd = np.min(drawdown_pct)
    calmar = -cagr/max_dd
    num_trades = events[abs(events['side'])>0].shape[0]
    avg_pnl = pnl.mean()
    hit_ratio = events[events['side']==1].shape[0]/num_trades
    
    summary = pd.Series([cagr,annualized_vol,sharpe,calmar, max_dd, num_trades, avg_pnl, hit_ratio], 
                       index=['Ann. Ret.','Ann. Vol.','Sharpe','Calmar','Max Drawdown','# Trades','Avg. PnL', 'Hit Ratio'])
    return summary


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
    

    
