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
    count = pd.Series(0, index=close_index[rel_idxs[0]:rel_idxs[-1]+1])
    for t_in, t_out in t1.iteritems():
        count.loc[t_in:t_out] += 1
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
    out['t1'] = avg_uniq
    return out


def get_ind_mat_label_uniqueness(ind_mat):
    """
    :param ind mat: (np.ndarray) indicator matrix (close idx, events) of cncrncy
    :return: (np.ndarray) matrix (events, close idx) of uniqueness
    """
    concurrency = ind_mat.sum(axis=1) # total concurrent close stamps per event
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

