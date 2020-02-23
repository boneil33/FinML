# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 18:16:47 2020

@author: Brendan
"""
import pandas as pd
import numpy as np
import datetime as dt
from numba import jit, prange


def _get_num_concurrent_events(close_index, t1):
    """
    :param close_index: (pd.DateIndex) close datestamps
    :param t1: (pd.Series) end datestamps indexed by entries
    :return: (pd.Series) number of live events indexed by closepx dates
    """
    t1 = t1.fillna(close_index[-1])
    rel_idxs = [close_index.searchsorted(t1.index[0]),
                close_index.searchsorted(t1.max())]
    count = pd.Series(0, index=close_index[rel_idxs[0]:rel_idxs[-1]+1], name='c')
    for t_in, t_out in t1.iteritems():
        count.loc[t_in:t_out] += 1
    return count


def get_num_conc_events_side(close_index, events):
    """
    Get number of concurrent events per side
    
    :param close_index: (pd.DateIndex) close datestamps
    :param events: (pd.DataFrame) events df with t1, side cols
    :return: (pd.DataFrame) number of live events per side indexed by closepx dates
    """
    events = events.copy(deep=True)
    events['t1'] = events.loc[:,'t1'].fillna(close_index[-1])
    rel_idxs = [close_index.searchsorted(events.index[0]),
                close_index.searchsorted(events['t1'].max())]
    close_rel_idx = close_index[rel_idxs[0]:rel_idxs[-1]+1]
    count = pd.DataFrame(np.zeros((len(close_rel_idx), 2)), index=close_rel_idx,
                         columns=['long','short'])
    for row in events.itertuples():
        if row.side>0:
            count.loc[row.Index:row.t1, 'long']+=1
        elif row.side<0:
            count.loc[row.Index:row.t1, 'short']+=1
    
    return count

    
def _get_average_uniqueness(t1, num_conc_events):
    """
    :param t1: (pd.Series) end stamps indexed by entries
    :param num_conc_events: (pd.Series) concurrent events per entry stamp
    :return: average uniqueness over event entry/exit
    """
    wgt = pd.Series(index=t1.index)
    for t_in, t_out in t1.iteritems():
        wgt.loc[t_in] = (1./num_conc_events.loc[t_in:t_out]).mean()
    
    return wgt


def get_max_concurrency(events, num_conc_events):
    """
    Generate max concurrency per event index
    
    :param events: (pd.DataFrame) events df with t1, side cols
    :param num_conc_events: (pd.DataFrame/Series) concurrent events per entry stamp, (per side)
    :return: (pd.Series) max previous live events (where side=thisrow side) indexed by events dates
    """
    if not isinstance(num_conc_events, pd.DataFrame):
        raise ValueError('num_conc_events must be DataFrame, got: ',
                         type(num_conc_events))
    
    out = pd.Series(0, index=events.index, name='maxsideconc')
    for row in events.itertuples():
        if row.side>0:
            out.loc[row.Index] = num_conc_events.loc[:row.Index, 'long'].max()
        elif row.side<0:
            out.loc[row.Index] = num_conc_events.loc[:row.Index, 'short'].max()
    
    return out


def get_events_avg_uniqueness(close, t1):
    """
    Wrapper to get average uniqueness for t1 series
    :param close: (pd.Series) close bar data
    :param t1: (pd.Series) end stamps indexed by entries
    :return: (pd.DataFrame) single column 'tw' average uniqueness for events
    """
    
    out = pd.DataFrame(index=close.index)
    num_conc_events = _get_num_concurrent_events(close.index, t1)
    avg_uniq = _get_average_uniqueness(t1, num_conc_events)
    out['tw'] = avg_uniq
    return out


def get_ind_mat(close_index, t1):
    """
    Get indicator matrix of live events per close index
    
    :param close: (pd.DateTimeIndex) close bar data
    :param t1: (pd.DateTimeIndex) end stamps indexed by entries
    :return: (np.ndarray) indicator matrix (close idx, events) of concurrency
    """
    c = np.array(close_index.values)
    start = np.array(t1.index.values)
    end = np.array(t1.values)
    ind_mat = ((c[:, None]>=start) & (c[:, None]<=end))*1
    
    return ind_mat
    
    
def get_ind_mat_label_uniqueness(ind_mat):
    """
    :param ind mat: (np.ndarray) indicator matrix (close idx, events) of cncrncy
    :return: (np.ndarray) matrix (events, close idx) of uniqueness
    """
    concurrency = ind_mat.sum(axis=1) # total concurrent events per close stamp
    uniqueness = ind_mat.T / concurrency
    return uniqueness


@jit(parallel=True, nopython=True)
def _bootstrap_loop(ind_mat, acc_concurrency): #needs testing
    """
    Runs the internal loop for sequential bootstrap. Generates average uniqueness
    :param ind_mat: (np.ndarray) (nxp) indicator matrix of concurrency,
        n = close index, p = t1 signal entries
    :param acc_concurrency: (np.array) (1xn) matrix of concurrency of 
        sequentially accumulated samples
    :return: (np.array) (1xp) average uniqueness of each event based on prev_conc
    """
    avg_unique = np.zeros(ind_mat.shape[1]) # label uniqueness
    
    #loop thru p cols
    for i in prange(ind_mat.shape[1]):
        prev_avg_uniq = 0
        num_elems = 0
        reduced_ind_mat = ind_mat[:,1]
        
        #loop thru n rows
        for j in range(len(reduced_ind_mat)):
            if reduced_ind_mat[j] > 0:
                new_elem = 1./(1.+acc_concurrency[j])
                avg_uniqueness = (prev_avg_uniq + num_elems + 
                                  new_elem) / (num_elems+1)
                num_elems+=1
                prev_avg_uniq = avg_uniqueness
        
        avg_unique[i] = avg_uniqueness
    return avg_unique


def seq_bootstrap(ind_mat, sample_length=None, random_state=np.random.RandomState()):
    """
    Generate sample for sequential bootstrap
    :param ind_mat: (np.ndarray) (nxp) indicator matrix of concurrency,
         n = close index, p = t1 signal entries
    :param sample_length: (optional int) number of samples we want to generate
    :return: (np.array) bootstrapped sample indices (of labels in p)
    """
    if sample_length is None:
        sample_length = ind_mat.shape[1]
        
    phi=[]
    acc_concurrency = np.zeros(ind_mat.shape[0]) #phi is empty, so 0 cncrncy
    while len(phi) < sample_length:
        avg_unique = _bootstrap_loop(ind_mat, acc_concurrency)
        prob = avg_unique / avg_unique.sum()
        choice = random_state.choice(range(ind_mat.shape[1]), p=prob)
        phi += [choice] #append latest sample
        #add newest sample's concurrency to accumulated cncrncy
        acc_concurrency += ind_mat[:, choice]
    
    return phi


def get_sample_weight(t1, close):
    """"
    Generate event sample weights by associated returns and concurrency
    
    :param t1: (pd.Series) end stamps indexed by entries
    :param close: (pd.Series) close bar data
    :return: (pd.Series) event absolute returns inv weighted by per period concurrency
    """
    ret = np.array(np.log(close).diff(1).fillna(0).values) # nx1
    ind_mat = get_ind_mat(close.index, t1) # n x m
    num_conc_events = ind_mat.sum(axis=1) # live events per close index (nx1)
    live_rets = ind_mat.transpose()*ret #live returns in rows per event (m x n)
    
    wgt = np.nansum(live_rets/num_conc_events[:, None].T, axis=1) # mx1
    
    #wgt[np.isnan(wgt)] = 0
    wgt*= wgt.shape/np.nansum(wgt)
    
    wgt = pd.Series(abs(wgt), index=t1.index, name='w')
    
    return wgt
    
    
if __name__=='__main__':
    c = pd.Series([1,2,3])
    e = pd.Series([1,1])
    ind = get_ind_mat(c, e)
    w = get_sample_weight(e,c)
    print(ind)