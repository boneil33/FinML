import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from Preprocessing.etf_trick import ETFTrick
import os
import itertools
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from Preprocessing.labeling import resample_close
from Preprocessing.feature_importance import get_pca_weights, get_pca_distances
from Research.fx_utils import fx_data_import


def pca_distance_loop(close, window, n_components, cluster_threshold, ccy_pairs, components_to_use=None):
    """
    todo: this should be more general, not just for FX pairs
    Main loop to generate timeseries of currency groups based on pca weight distances
    
    :param close: (pd.DataFrame) close prices
    :param window: (int) length of window used to compute pca
    :param n_components: (int) number of components to return weights on
    :param cluster_threshold: (float) distance threshold below which we accept pairs
    :param components_to_use: (np.array) component numbers to use (e.g. 1,2)
    :return: (pd.DataFrame) timeseries indicator of distinct below threshold pairs (e.g. BRLMXN)
    """
    close = close.dropna(how='any')
    # full timeseries
    distance_group_ts = pd.DataFrame(np.zeros((close.shape[0], len(ccy_pairs))), 
                                     index=close.index, columns=ccy_pairs)
    # single row placeholder
    distance_group = pd.Series(np.zeros(len(ccy_pairs)), index=ccy_pairs)
    # idx will be the index at which the window ends
    # idx-window wlil be the index at which the window starts
    # idx-window*2:idx-window is where we use some data to compute rolling mean+std
    sample_window = window*2
    for idx in np.arange(sample_window,close.shape[0],window):
        sub_close = close.iloc[idx-sample_window:idx]
            
        components = get_pca_weights(sub_close, window, n_components=n_components)
        # reset at 0
        
        distances = get_pca_distances(components)
        # usd-pair distances to non_usd pairs
        for j in np.arange(close.shape[1]):
            for i in np.arange(j+1,close.shape[1]):
                if distances.iloc[i, j] <= cluster_threshold:
                    row, col = close.columns[i], close.columns[j]
                    distance_group[col[-3:]+row[-3:]] = 1
                    
        # copy the indictators forward by the window length (we want the same
        # indicators until we recalculate them)
        for i in range(window):
            if idx+i == distance_group_ts.shape[0]:
                break
            distance_group_ts.iloc[idx+i] = distance_group[ccy_pairs].values
        distance_group*=0
        
    return distance_group_ts


def get_nonusd_pair_data(close, yields, nonusd_cols):
    """
    Get nonusd ccy pair closes (e.g. EURGBP) from USD denom closes
    all yields from bbg (e.g. EURI3M, or CHFI3M) are nonUSD yields
    """
    nonusd_close = pd.DataFrame(index=close.index)
    nonusd_yields = pd.DataFrame(index=close.index)
    for col in nonusd_cols:
        col1 = col[:3]
        col2 = col[-3:]
        c1 = close['USD'+col1]
        c2 = close['USD'+col2]
        y1 = yields['USD'+col1]
        y2 = yields['USD'+col2]
        nonusd_close[col] = c2.divide(c1)
        nonusd_yields[col] = y1.subtract(y2)
    
    return nonusd_close, nonusd_yields
    

def get_nonusd_pairs(close_cols):
    """
    Get the nonUSD pairs
    """
    non_usd = [s[-3:] for s in close_cols]
    combos = itertools.combinations(non_usd, 2)
    ccy_pairs = [i[0]+i[1] for i in combos]
    
    return ccy_pairs


if __name__=='__main__':
    full_data = fx_data_import()
    
    c1 = 'USDSEK'
    c2 = 'USDINR'
    
    sub_data = full_data[[c1,c2]].dropna(how='any')
    weights = pd.DataFrame(np.ones(sub_data.shape), index=sub_data.index,
                        columns=sub_data.columns)
    weights[c2] = -1.
    inv_rates = 1./sub_data
    carry = pd.DataFrame(np.zeros(sub_data.shape), index=sub_data.index,
                         columns=sub_data.columns)
    # shift close data one forward to create open prices
    # shift inverse data one forward to create last close
    etf = ETFTrick(sub_data.shift(1), sub_data, weights, carry, inv_rates.shift(1))
    
    etf_s = etf.get_etf_series()
    
    sub_data[c1+c2] = sub_data[c1].divide(sub_data[c2])
    fix, ax = plt.subplots()
    ax.plot(etf_s)
    ax.plot(sub_data[c1+c2]/sub_data[c1+c2].iloc[0])
    ax.legend(['ETF',c1+c2])
    plt.show()
