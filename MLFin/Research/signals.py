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


def lookback_zscore(close, lookback, vol_lookback, log_ret=True, center=True, vs_ma=False):
    """
    Generate lookback-period return zscores
    
    :param close: (pd.DataFrame) close prices
    :param lookback: (int) periods to look back
    :param vol_lookback: (int) rolling window size
    :return: (pd.DataFrame) rolling lookback zscores
    """
    if log_ret:
        close = close.apply(np.log)
    
    if vs_ma:
        rets = close.subtract(close.rolling(lookback).mean()).dropna()
    else:
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


def cusum_sym_filter(srs, threshold_pct):
    """
    Symmetric cusum filter, adapted to trigger when series is over threshold but has a one day reversal
    :param srs: (pd.Series) some series of values
    :param threshold: (float) percentage of series mean at which to trigger signal
    :returns: (pd.Series) signed signals indexed by trigger date
    """
    dates = []
    signals = []
    thresh = srs.mean()*threshold_pct
    diffs = srs.diff(1).dropna()
    sPlus = 0
    sMinus = 0
    for idx in diffs.index[1:]:
        val = diffs.loc[idx]
        
        sPlus = max(0, sPlus+val)
        sMinus = min(0, sMinus+val)
        if val < 0:# and sPlus > thresh:
            sPlus = 0
            dates.append(idx)
            signals.append(-1)#srs.loc[idx])
        elif val > 0:# and sMinus < -thresh:
            sMinus = 0
            dates.append(idx)
            signals.append(1)#srs.loc[idx])
    
    return pd.Series(signals, index=dates, name='CusumFilter')


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
                  trgt=None, model_resids=None, max_signals=None, even_weight=False,
                 exit_pct=False):
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
    events, df0 = get_events(events0, close, trgt, pt_sl=pt_sl, model_resids=model_resids, exit_pct=exit_pct)
    
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
    
    return events, df0


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
        # we open position at the close, should be no pnl contribution, just the TC
        pnl_df.loc[row.Index] += -tc - close_diff.loc[row.Index]*row.side*row.size
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
        exposures.loc[row.Index] -= 1
    
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


def createPerfSummaryDFs(summaries):
    idx = pd.IndexSlice
    summaries_df = pd.DataFrame(summaries).T
    summaries_df.index.set_names(['max_signals', 'resid_lookback', 'entry_z', 'exit_z'], inplace=True)
    summaries_df = summaries_df.reset_index()
    summaries_df.loc[:, 'Avg. Trade Days'] = summaries_df.loc[:, 'Avg. Trade Days'].dt.days
    #summaries_df.loc[:, ['entry_z', 'exit_z']] = summaries_df.loc[:, ['entry_z', 'exit_z']]
    summaries_df.set_index(['max_signals', 'resid_lookback', 'entry_z', 'exit_z'], inplace=True)
    summaries_df_noindex = summaries_df.reset_index()
    conv_cols = summaries_df_noindex.select_dtypes(include='object').columns
    summaries_df_noindex.loc[:, conv_cols] = summaries_df_noindex.loc[:, conv_cols].astype(float, errors='ignore')
    return summaries_df, summaries_df_noindex


from mpl_toolkits.axes_grid1 import make_axes_locatable
def get_event_bins(event_row, bins):
    score = event_row['entry_score']
    if score < bins[0]:
        return 0
    elif score > bins[-1]:
        return len(bins)
    else:
        for i in range(len(bins)-1):
            if score >= bins[i] and score < bins[i+1] :
                return i+1
    raise ValueError('Did not fall in a bin')


def _get_tc(events, tc_pct, rebal_cost):
    """
        Compute rebal costs:
        1. apply to events['ret']
        2. compute the 'live' rebal costs, i.e. the rebalances that occur as a trade is live
            - multiply costs by live sizes
    
    :param events: (pd.DataFrame) final events with ret and size column
    :param rebal_cost: (pd.Series) rebalance costs of $1 etf over life of etf
    :return: events with updated ret, rebal_cost updated series
    """
    ret_events = events.copy(deep=True)
    rebal_cost = rebal_cost.rename('cost')
    ret_cost = rebal_cost.copy(deep=True).to_frame()
    ret_cost['abs_size'] = 0.0
    for row in events.itertuples():
        # apply cost to ret_events, in inv vol scaled terms (since that is scale of ret)
        ret_events.loc[row.Index, 'ret'] += ( rebal_cost.loc[row.Index:row.t1].sum() - 
                                             2.*tc_pct )*row.size/row.trgt
                                                
        
        # compute 'live' rebal costs, in $1 terms, inclusive of trade dates to be conservative
        ret_cost.loc[row.Index:row.t1, 'abs_size'] += row.size
    
    ret_cost['strat_cost'] = ret_cost['abs_size']*ret_cost['cost'] 
    return ret_events, ret_cost['strat_cost']


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
    pnl_index = pnl_index.loc[events.index[0]:last_date]
    years_live = (last_date-events.index[0]).days/365.25
    cagr = np.power(pnl_index.iloc[-1]/pnl_index.iloc[0],1/years_live)-1.
    
    log_returns = pnl_index.apply(np.log).diff(1).dropna()
    log_m_returns = pnl_index.copy(deep=True).asfreq('BM').apply(np.log).diff(1).dropna()
    vol_d = log_returns.std()
    vol_m = log_m_returns.std()
    # assumes daily close data
    annualized_dvol = vol_d*np.sqrt(252)
    sharpe_d = cagr/annualized_dvol
    # monthly and account for autocorr
    acf = tsa.stattools.acf(log_m_returns, nlags=11)
    annualized_mvol = vol_m*np.sqrt(12+2*sum([(12-i)*acf[i] for i in range(1,12)]))
    sharpe_m = cagr/annualized_mvol
    
    drawdown_pct = pnl_index.divide(pnl_index.expanding(0).max())-1.
    max_dd = np.min(drawdown_pct)
    calmar = -cagr/max_dd
    num_trades = events[abs(events['side'])>0].shape[0]
    
    # update ret for at least execution tc, 2x in and out 
    
    avg_pnl = events['ret'].mean()
    long_ratio = events[events['side']==1].shape[0]/num_trades
    hit_ratio = events[events['ret']>0].shape[0]/num_trades
    avg_trade_days = (events['t1']-events.index).mean()
    
    summary = pd.Series([cagr, annualized_dvol, annualized_mvol, sharpe_d, sharpe_m, calmar, max_dd, num_trades, avg_pnl, 
                         long_ratio, hit_ratio, avg_trade_days, events.index.min(), events.t1.max()], 
                       index=['Ann. Ret.', 'Ann. Vol.', 'Ann. Vol. (m)', 'Sharpe', 'Sharpe (m)', 
                              'Calmar', 'Max Drawdown', '# Trades',
                              'Avg. PnL', 'Long%Signals', 'Hit Ratio', 'Avg. Trade Days', 'First Entry', 'Last Exit'])
    return summary
   

def draw_signal_response(events, bstart=-2., bend=2., bwidth=0.25):
    # weight the returns by avg uniqueness and take away side (so want shorts to return negative)
    events['tw'] = get_events_avg_uniqueness(events['t1'], events['t1'])
    events['wgt_ret_noside'] = -events['ret']*events['side']*events['tw']

    bwidth = 0.25
    bstart = -2.
    bend = 2.25
    bins = np.arange(bstart, bend, bwidth) #[-2., -1.5, -1., -.75, -.5, -.25, 0, .25, .5, .75, 1., 1.5, 2.]
    
    # get integer bins for signals
    events['bin'] = events.apply(lambda x: get_event_bins(x, bins), axis=1)
    fig, ax = plt.subplots(figsize=(8,6), dpi=300)
    ax = sns.boxplot('bin', 'wgt_ret_noside', data=events, ax=ax, notch=True)

    ticks = ax.get_xticks()
    ax.set_xticks(ticks+0.5)
    ax.set_xticklabels(bins)
    # get index of 0
    idx0 = bins.searchsorted(0)
    ax.axvline(idx0+0.5, c='r', ls='-.', lw=0.5) #0 will be shifted right 0.5

    divider = make_axes_locatable(ax)
    axHist = divider.append_axes("bottom", 1.2, pad=0.1)#, sharex=ax)
    adj_bins = bins
    adj_bins = np.insert(adj_bins, 0, bstart-bwidth)
    adj_bins = np.append(adj_bins, bend+bwidth)
    axHist.hist(events['bin'], bins=range(len(bins)+2), rwidth=0.9)

    axHist.set_xticks(ticks+1.0)
    axHist.set_xticklabels(bins)
    ax.set_xlim(-0.5, len(bins)+0.5)
    axHist.set_xlim(0, len(bins)+1.0)
    axHist.set_xlabel('entry_z')
    ax.set_title('Signal Response')

    plt.show()
    
    
def run_resid_backtest(closes, weights, carry, resids, resid_lookback_diff, vol_lookback, entry_threshold, pt_sl, vertbar,
                       max_signals=1, even_weight=True, log_ret=False, vs_ma=False, rebal_freq=None, plot=True, tc_pct=0.0, 
                       roll_cost=None, cusum_pct=None, exit_pct=False, window=3.):
    """
        Run a backtest based on residual reversion. Plot the resulting positions and MTM pnl, and show the stats summary
    :param resids: (pd.Series) of residuals
    :param resid_lookback_diff: (int) trading days to compute difference to generate zscore signals
    :param vol_lookback: (int) trading days to compute lookback vol for zscore signals
    :param entry_threshold: (float) zscore entry
    :param rebal_freq: (pandas date freq or None) how often to rebalance portfolio outside of weight changes
        in other words, how often pnl is reinvested
    :param tc_pct: (float) transaction cost assumption 1e-4 = 1bp
    :param roll_costs: (pd.Series) roll costs assumption indexed by roll dates
    """
    # generate ETF synthetic series

    tr = ETFTrick(closes.shift(1), closes, weights, carry, rebal_freq=rebal_freq)
    trs, etf_inter_data = tr.get_etf_series(return_data=True)
    
    # need to reindex the residuals to days where my trs are defined (i.e. drop NaN days)
    resids = resids.reindex(trs.index)

    # rolling change zscores of the aggregated residuals
    # TODO: maybe we should consider change versus a moving average rather than a specific lookback?
    zscore_resids = lookback_zscore(resids, lookback=resid_lookback_diff, vol_lookback=vol_lookback, 
                                    log_ret=log_ret, vs_ma=vs_ma)
    
    # generating over threshold .75 z-score signals for reversion
    signals = zscore_signal(zscore_resids, threshold=entry_threshold, signal_type='Reversion').rename('signal')

    # cusum reversal too
    if cusum_pct is not None:
        cusum_signals = cusum_sym_filter(resids, cusum_pct)
        # full signal from cusum and zscore
        full_signals = signals.to_frame().join(cusum_signals).fillna(0).apply(lambda x: x['signal'] if x['signal']==x['CusumFilter'] 
                                                                      else 0, axis=1)
    else:
        full_signals = signals
    
    # generate initial events DataFrame for this reversion strategy
    events0, df0 = zscore_sizing(full_signals, trs, vertbar=vertbar, pt_sl=pt_sl, 
                                 max_signals=max_signals, even_weight=even_weight, model_resids=zscore_resids,
                                exit_pct=exit_pct)
    
    events = generate_pnl(events0, trs, pct_change=True)
    
    # tc cost on $1 (i.e. 1bp = 1e-4), weights_diff_delev is absolute change in weights
    rebal_cost_srs = -tc_pct*etf_inter_data['weights_diff_delev'].squeeze().rename('weights')
    
    if roll_cost is not None:
        # add tc cost on $1 rolled
        total_rebal_cost_srs = rebal_cost_srs.to_frame().join(roll_cost.rename('roll_cost'), how='outer').fillna(0.0)
        total_rebal_cost_srs['total_cost'] = total_rebal_cost_srs['weights']+total_rebal_cost_srs['roll_cost']
    else: 
        total_rebal_cost_srs=rebal_cost_srs.rename('total_cost').to_frame()
    
    # update events['ret'] for tc and calculate full cost to apply to mtm_pnl
    events, rebal_cost_strat = _get_tc(events, tc_pct, total_rebal_cost_srs['total_cost'])
    
    mtm_pnl = generate_mtm_pnl(events, trs, log_diff=True, tc=tc_pct) # subtract tc from return on open and close, doesn't use ret
    rebal_cost_strat = rebal_cost_strat.reindex(mtm_pnl.index, fill_value=0.0).rename('strat')
    
    pnl_index = generate_pnl_index(mtm_pnl, rebal_cost_strat)
    exposures = generate_exposures(events, trs)
    if plot:
        # compute data for rolling sharpe
        window_m = int(window*12)
        rolling_pct = pnl_index.apply(np.log).diff(252*window)
        cagr_srs = np.power(pnl_index.pct_change(int(252*window))+1., 1/window)-1.

        log_m_returns = pnl_index.copy(deep=True).asfreq('BM', method='ffill').apply(np.log).diff(1).dropna()

        vol_m_srs = log_m_returns.rolling(window_m).std()

        # monthly and account for autocorr
        acorr_scale = log_m_returns.rolling(window_m).apply(lambda x: np.sqrt(12 + 
                                                              2*sum([(12-i) * x.autocorr(lag=i) for i in range(1,12)])), 
                                                              raw=False).fillna(1.)
        mvol_srs_ann = vol_m_srs * acorr_scale
        sharpe_m = cagr_srs.asfreq('BM', method='ffill') / mvol_srs_ann
        fig, ax = plt.subplots(figsize=(6,7), nrows=3, dpi=300)
        exposures.plot(ax=ax[0])
        pnl_index.plot(ax=ax[1])
        sharpe_m.plot(ax=ax[2])
        ax[0].set_title("Exposure in Basket")
        ax[1].set_title(r"Daily MTM Equity Curve ({0}bp TC)".format(np.round(tc_pct*1e4, 1)))
        ax[2].set_title(r"Rolling {0}yr Sharpe {1}bp TC".format(window, np.round(tc_pct*1e4, 1)))
        fig.tight_layout()
        plt.show()
    perf_summary = generate_perf_summary(events, trs, tc_pct=tc_pct, rebal_cost=rebal_cost_strat)
    
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
    

    
