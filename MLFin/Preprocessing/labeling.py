# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 17:27:14 2020

@author: Brendan
"""
import pandas as pd
import numpy as np
import datetime as dt


def resample_close(close, period):
    out = close.copy(deep=True).asfreq(period)
    return out


def get_lookback_vol(close, lookback, volwindow=100, ewma=True):
    rets = close.apply(np.log).diff(lookback).dropna()
    if ewma:
        vols = rets.ewm(span=volwindow).std().dropna()
    else:
        vols = rets.rolling(volwindow).std().dropna()
    vols = vols.rename('Vol')
    return vols


def get_t1(s, close, vertbar=30):
    """
    :param s: (pd.int64index,pd.datetimeindex) entry signal dates index
    :param close: (pd.Series) close prices
    :param vertbar: (int) vertical bar distance from t0 (cal day)
    :return: (pd.Series) timeouts
    """
    if isinstance(s, pd.Series):
        s = s.index
        
    #closes vertbar bars from index
    if not isinstance(s, pd.core.indexes.datetimes.DatetimeIndex):
        t1 = close.index.searchsorted(s+vertbar)
    else:
        t1 = close.index.searchsorted(s+pd.Timedelta(days=vertbar))
    #t1 = t1[t1<close.shape[0]]
    t1[t1==close.shape[0]] = close.shape[0]-1
    t1 = pd.Series(close.index[t1], index=s[:t1.shape[0]]).dropna()
    return t1


def _apply_pt_sl(events, close, pt_sl=(1,1)):
    """
    Applies tp/sl and updates t1 to include horizontal bar hits
    
    :param events: (pd.DataFrame) index=entry, t1=timeouts, trgt=trailing vol
        signal, side=trade side
    :param close: (pd.Series) close prices
    :param tp_sl: (list) 2-tuple take profit/stop loss
    :return: (pd.DataFrame) adds t1, sl, tp to events
    """
    if 'side' not in events.columns:
        events['side'] = 1
    out = events['t1'].copy(deep=True).to_frame()
    out['tp'] = pd.NaT
    out['sl'] = pd.NaT
    tp = pt_sl[0]*events['trgt'] #tgt take profit in vol units
    sl = -pt_sl[1]*events['trgt']
    for loc, t1 in events['t1'].fillna(close.index[-1]).iteritems():
        df0 = close.loc[loc:t1] #closes from trade entry to time out
        #returns from loc to each time loc:t1
        df0 = (df0.divide(close.loc[loc])-1.)*events.loc[loc,'side']
        out.loc[loc, 'tp'] = df0[df0>tp.loc[loc]].index.min()#earliest tp
        out.loc[loc, 'sl'] = df0[df0<sl.loc[loc]].index.min()
    return out


def _apply_pt_sl_vs_model(events, close, model_resids, pt_sl=(1,1), exit_pct=False):
    """
    Applies tp/sl based on model residual zscores, pt_sl in terms of zscore
        positive numbers are on the same side of 0 as the zscore
        thus negative values are more aggressive for take profit
    e.g. pt = 0.25, sl = 2.0, long signal generated at -1.0 (not provided)
        we would stop out at -side*2.0 (-2) and tp at at 0.25*-side (0.25)
    :param events: (pd.DataFrame) index=entry, t1=timeouts, trgt=trailing vol
        signal, side=trade side
    :param close: (pd.Series) close prices
    :param model_resids: (pd.Series) model_residual series zscores
    :param tp_sl: (list) 2-tuple take profit/stop loss
    :param exit_pct: (bool) use tp_sl as percent of entry z-score
    :return: (pd.DataFrame) adds t1, sl, tp to events
    """
    if 'side' not in events.columns:
        raise ValueError('Side must be provided for model pt_sl generation')
    
    out = events['t1'].copy(deep=True).to_frame()
    out['tp'] = pd.NaT
    out['sl'] = pd.NaT
    if exit_pct:
        if 'entry_score' not in events.columns:
            raise ValueError('If exit_pct is true need to have the entry zscore')
        out['tp_val'] = abs(events.loc[:, 'entry_score']) * pt_sl[0]
        out['sl_val'] = abs(events.loc[:, 'entry_score']) * pt_sl[1]
    else:
        out['tp_val'] = pt_sl[0]
        out['tp_val'] = pt_sl[1]
    
    # do we use pt_sl as a pct of the entry signal or just a float

    for loc, t1 in events['t1'].fillna(close.index[-1]).iteritems():
        # residuals at each point from loc:t1
        df0 = model_resids.loc[loc:t1]
        if events.loc[loc, 'side']==1:
            out.loc[loc, 'tp'] = df0[df0>-out.loc[loc, 'tp_val']].index.min() # earliest tp
            out.loc[loc, 'sl'] = df0[df0<-out.loc[loc, 'sl_val']].index.min()
        else:
            out.loc[loc, 'tp'] = df0[df0<out.loc[loc, 'tp_val']].index.min() # earliest less than thresh
            out.loc[loc, 'sl'] = df0[df0>out.loc[loc, 'sl_val']].index.min() # earliest greater than
    
    out.drop(['tp_val', 'sl_val'], axis=1, inplace=True)
    return out
    
    
def _get_barrier_touched(df0, events):
    """
    :param df0: (pd.DataFrame) contains returns and targets
    :param events: (pd.DataFrame) orig events df containing tp and sl cols
    :return: (pd.DataFrame) with returns, targets and labels, including 0 if
        timeout hit first
    """
    store = []
    for t_in, vals in df0.iterrows():
        ret = vals['ret']
        trgt = vals['trgt']
        
        pt_hit = ret > trgt*events.loc[t_in, 'tp']
        sl_hit = ret < -trgt * events.loc[t_in, 'sl']
        if ret > 0 and pt_hit:
            store.append(1)
        elif ret < 0 and sl_hit:
            store.append(-1)
        else:
            store.append(0)
    df0['bin'] = store
    return df0


def get_events(events, close, trgt, pt_sl=(1,1), minret=0.000001, model_resids=None, exit_pct=False):
    """
    :param events: (pd.DataFrame) index=entry, t1=timeouts, side=tradeside
    :param close: (pd.Series) of close prices
    :param trgt: (pd.Series) time depdt target sizings (trailing vols)
    :param model_resids: (pd.Series) model zscores
    :param exit_pct: (bool) whether to compute exit signals as percentage of entry signal
    :return: (pd.DataFrame) index=entry, t1=exit tp or sl, side=tradeside
    """
    side_pred = True
    trgt = trgt.reindex(events.index)
    trgt = trgt[trgt>minret]
    if 't1' not in events.columns:
        events['t1'] = pd.NaT
    if 'side' not in events.columns:
        side_pred = False
        events['side'] = 1
        pt_sl = [pt_sl[0], pt_sl[0]]
    events['trgt'] = trgt
    events = events.dropna(subset=['trgt'])
    if model_resids is not None:
        events['entry_score'] = model_resids
        df0 = _apply_pt_sl_vs_model(events, close, model_resids, pt_sl, exit_pct=exit_pct)
    else:
        df0 = _apply_pt_sl(events, close, pt_sl)
    
    events.loc[:, 't1'] = df0.dropna(how='all').min(axis=1)
    
    if not side_pred:
        events = events.drop('side', axis=1)
    if exit_pct:
        # exit is a percentage of the entry signal
        if 'entry_score' not in events.columns:
            raise ValueError('if exit_pct is true need entry_score column')
        events.loc[:, 'tp'] = pt_sl[0]*events['entry_score']
        events.loc[:, 'sl'] = pt_sl[1]*events['entry_score']
    else:
        events.loc[:, 'tp'] = pt_sl[0]
        events.loc[:, 'sl'] = pt_sl[1]
    return events, df0


def get_bins(events, close):
    """
    Generate outcomes for events
    :param events: (pd.DataFrame) index=entry, t1=exit, trgt, side (optional)
        implies side generated by another logic, if side in events (0,1)
        if side not in events (-1,1) by price action
    :param close: (pd.Series) close prices
    :return: (pd.DataFrame) meta-labeled events
    """
    events_ = events.copy(deep=True).dropna(subset=['t1'])
    # in case t1 doesnt have to have same sample as entries
    all_dates = events_.index.union(other=events_['t1'].values).drop_duplicates()
    log_close = close.reindex(all_dates, method='bfill').apply(np.log)
    df0 = pd.DataFrame(index=events_.index)
    # use log return or shorts will be skewed
    df0['ret'] = (log_close.loc[events['t1'].values].values -
                   log_close.loc[events.index].values)
    df0['trgt'] = events_['trgt']
    
    #metalabel
    if 'side' in events.columns:
        df0['ret'] = df0['ret'] * events_['side']
    df0 = _get_barrier_touched(df0, events)
    
    if 'side' in events.columns:
        df0.loc[df0['ret'] <= 0, 'bin'] = 0
        df0['side'] = events_['side']
    #back to arithmetic returns
    df0['ret'] = np.exp(df0['ret'])-1
    
    return df0



