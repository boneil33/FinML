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


def generate_pnl(events, close_tr, pct_change=True):
    """
    Generate pnl series from events. Assumes close_tr and close used to 
        generate events df have same index
    
    Alternatively I should use total return index throughout this module
        
    :param events: 'events' dataframe with t1, side, size, trgt
    :param close_tr: (pd.Series) single security total return index
    :return: (pd.Series) strategy pnl series, inversely scaled by the trailing vol
    """
    if pct_change:
        events['ret'] = close_tr.loc[events['t1']].values/close_tr.loc[events.index].values-1.
    else:
        events['ret'] = close_tr.loc[events['t1']].values-close_tr.loc[events.index].values
    events['ret']*= events['side']*events['size']/events['trgt']
    
    return events
    

def generate_mtm_pnl(events, close_tr, log_diff=True, tc=0.0):
    """
    Generate daily mark to market pnl from events.
    todo: paralellize this!
    
    :param events: 'events' dataframe with t1, side, size, trgt
    :param close_tr: (pd.Series) single security total return index
    :param tc: (float) transaction cost in return units, e.g. 1bp = 1e-4
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
    pnl_df = pd.Series(0, index=close_tr.index, name='strat')
    for row in events.itertuples():
        pnl_df.loc[row.Index:row.t1] += close_diff.loc[row.Index:row.t1]*row.side*row.size#*row.trgt
        pnl_df.loc[row.Index] = -tc # we open position at the close, should be no pnl
        pnl_df.loc[row.t1] += -tc
    
    #re-exponentiate
    if log_diff:
        pnl_df = pnl_df.apply(np.exp)-1.
    
    return pnl_df


def generate_exposures(events, close):
    """
    Generate exposure per day
        exposure is measured at the open
    """
    exposures = pd.Series(0, index=close.index)
    for row in events.itertuples():
        exposures.loc[row.Index:row.t1] += row.side*row.size
        exposures.loc[row.Index] = 0
    
    return exposures


def generate_pnl_index(mtm_pnl, rebal_cost=None):
    """
    Generate strategy total return index from mtm_pnl
    
    Effectively just treat it as carry and use ETFTrick.get_etf_series()
    :param mtm_pnl: (pd.Series) $pnl series from generate_mtm_pnl function
    :param rebal_cost: (pd.Series) $daily cost of rebalancing
    """
    df1 = pd.DataFrame(1, index=mtm_pnl.index, columns=['strat'])
    
    carry_of_strat = mtm_pnl.rename('strat').to_frame()
    if rebal_cost is not None:
        carry_of_strat = (mtm_pnl.add(rebal_cost)).rename('strat').to_frame()
    
    trick = ETFTrick(df1.shift(1), df1, df1, carry_of_strat)
    index_pnl = trick.get_etf_series()
    return index_pnl

    
def generate_perf_summary(events, close_tr, tc_pct=0.0, rebal_cost=None):
    """
    Function to generate CAGR, vol, sharpe, calmar, max drawdown, # trades, avg pnl per trade, hit ratio
    
    :input events: (pd.DataFrame) 'events' dataframe with t1, side, size, trgt
    :input close_tr: (pd.Series) total return series of underlying product
    :return: (pd.DataFrame) summary of pnl attributes
    """
    pnl = generate_mtm_pnl(events, close_tr, log_diff=True, tc=tc_pct)
    if rebal_cost is not None: 
        pnl_index = generate_pnl_index(pnl, rebal_cost)
    else:
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
    
    # update ret for at least execution tc, 2x in and out
    events.loc[:, 'ret'] -= 2*tc_pct*events.loc[:, 'size']/events.loc[:, 'trgt']
    avg_pnl = events['ret'].mean()
    long_ratio = events[events['side']==1].shape[0]/num_trades
    hit_ratio = events[events['ret']>0].shape[0]/num_trades
    avg_trade_days = (events['t1']-events.index).mean()
    
    summary = pd.Series([cagr,annualized_vol,sharpe,calmar, max_dd, num_trades, avg_pnl, 
                         long_ratio, hit_ratio, avg_trade_days], 
                       index=['Ann. Ret.','Ann. Vol.','Sharpe','Calmar','Max Drawdown','# Trades',
                              'Avg. PnL', 'Long%Signals', 'Hit Ratio', 'Avg. Trade Days'])
    return summary


def run_resid_backtest(closes, weights, carry, resids, resid_lookback_diff, vol_lookback, entry_threshold, pt_sl, vertbar,
                       max_signals=1, even_weight=True, log_ret=False, rebal_freq=None, plot=True, tc_pct=0.0):
    """
        Run a backtest based on residual reversion. Plot the resulting positions and MTM pnl, and show the stats summary
    :param resids: (pd.Series) of residuals
    :param resid_lookback_diff: (int) trading days to compute difference to generate zscore signals
    :param vol_lookback: (int) trading days to compute lookback vol for zscore signals
    :param entry_threshold: (float) zscore entry
    :param rebal_freq: (pandas date freq or None) how often to rebalance portfolio outside of weight changes
        in other words, how often pnl is reinvested
    """
    # generate ETF synthetic series

    tr = ETFTrick(closes.shift(1), closes, weights, carry, rebal_freq=rebal_freq)
    trs, etf_inter_data = tr.get_etf_series(return_data=True)
    
    # need to reindex the residuals to days where my trs are defined (i.e. drop NaN days)
    resids = resids.reindex(trs.index)

    # rolling change zscores of the aggregated residuals
    # TODO: maybe we should consider change versus a moving average rather than a specific lookback?
    zscore_resids = lookback_zscore(resids, lookback=resid_lookback_diff, vol_lookback=vol_lookback, log_ret=log_ret)
    
    # generating over threshold .75 z-score signals for reversion
    signals = zscore_signal(zscore_resids, threshold=entry_threshold, signal_type='Reversion')

    # generate initial events DataFrame for this reversion strategy
    events0, df0 = zscore_sizing(signals, trs, vertbar=vertbar, pt_sl=pt_sl, 
                                 max_signals=max_signals, even_weight=even_weight, model_resids=zscore_resids)
    events = generate_pnl(events0, trs, pct_change=True)
    
    # tc cost on $1 (i.e. 1bp = 1e-4)
    rebal_cost_srs = -tc_pct*etf_inter_data['weights_diff_delev']
    
    mtm_pnl = generate_mtm_pnl(events, trs, tc_pct) # subtract tc from return on open and close
    rebal_cost_srs = rebal_cost_srs.squeeze().reindex(mtm_pnl.index, fill_value=0.0).rename('strat')
    
    pnl_index = generate_pnl_index(mtm_pnl, rebal_cost_srs)
    exposures = generate_exposures(events, trs)
    if plot:
        fig, ax = plt.subplots(figsize=(6,6), nrows=3, dpi=300)
        exposures.plot(ax=ax[0])
        pnl_index.plot(ax=ax[1])
        trs.plot(ax=ax[2])
        ax[0].set_title("Exposure in long USDCAD vs Basket")
        ax[1].set_title(r"Daily MTM PnL ({0}bp TC)".format(int(tc_pct*1e4)))
        fig.tight_layout()
        plt.show()
    perf_summary = generate_perf_summary(events, trs, tc_pct=tc_pct, rebal_cost=rebal_cost_srs)
    
    return events, df0, pnl_index, exposures, trs, perf_summary



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
    

    
