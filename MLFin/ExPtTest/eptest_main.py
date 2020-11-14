# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:24:31 2020

@author: Brendan
"""

import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
import statsmodels.api as sm
import statsmodels.tsa as tsa

from sklearn.linear_model import Lasso, Ridge, LassoCV, RidgeCV, LogisticRegression, LinearRegression
from sklearn.model_selection import TimeSeriesSplit, train_test_split, GridSearchCV

from Research.signals import zscore_signal

CT_SWAP_MAP = {
    'TU': ['2y'],
    'FV': ['5y'],
    'TY': ['7y'],
    'US': ['swap_US'],
    'WN': ['25y']
}

CT_SIZES = {
    'TU': 2e5,
    'FV': 1e5,
    'TY': 1e5,
    'US': 1e5,
    'WN': 1e5
}

COT_COLS = ['Com', 'NonCom', 'NonRep', 'AM', 'LevFunds', 'Dealers', 'OtherRep']
PRICE_DUR_COLS = ['PX_LAST', 'FUT_EQV_DUR_NOTL', 'OPEN_INT']
def get_interp_swaps(swaps_raw):
    swaps = swaps_raw.copy(deep=True)
    us_gap_dt = dt.date(2015,3,11)
    swaps_len = swaps.shape[0]
    rel_idx = swaps.loc[us_gap_dt:].index
    len_post = len(rel_idx)
    the_roll = swaps.loc[us_gap_dt, '15y'] - swaps.loc[us_gap_dt, '20y']
    the_roll_arr = np.hstack([[0.]*(swaps_len-len_post), [the_roll]*len_post])
    
    # construct approximate weights for relevant swap for US
    w_20y = np.hstack([[0.]*(swaps_len-len_post), [np.interp(i, [0, len_post], [1, 0]) for i in range(1, len_post+1)]])
    w_15y = np.hstack([[1.]*(swaps_len-len_post), list(reversed(w_20y))])
    
    # interpolate between 20y and 15y to adjust for US contract issues 
    weights = pd.DataFrame([[a,b] for a,b in zip(w_20y, w_15y)], index=swaps.index, columns=['w_20y', 'w_15y'])
    swaps['swap_US'] = swaps.loc[:, ['20y', '15y']].join(weights).apply(lambda x: x['20y'] * x['w_20y']
                                                    + x['15y'] * x['w_15y'], axis=1)
    # apply simple roll to smooth returns for US
    swaps.loc[:, 'swap_US'] = swaps.loc[:, 'swap_US'] + the_roll_arr
    
    # make simple 25y swap
    swaps['25y'] = (swaps['30y'] + swaps['20y']) / 2.
    #swaps.loc[:, ['15y', 'swap_US', '20y']].plot()
    #swaps.loc[dt.date(2015,3,10):, ['15y','swap_US']]
    return swaps
    

def rename_legacy_cols(col):
    col_name = col.lower()
    if 'non-commercial' in col_name: 
        return 'NonCom'
    elif 'net commercial' in col_name:
        return 'Com'
    elif 'non-reportable' in col_name:
        return 'NonRep'
    elif 'net total futures' in col_name:
        return 'Total'
    else:
        raise ValueError(r"colname: {0} doesnt map".format(col))
    

def rename_tff_cols(col):
    colmap = {'Asset Manager Institutional Net Total/Futures': 'AM',
              'Leveraged Funds Net Total/Futures': 'LevFunds',
              'Dealer Intermediary Net Total/Futures': 'Dealers',
              'Reportables Net Total/Futures': 'OtherRep',
              'Other Reportables Net Total/Futures': 'OtherRep'}
    if col in colmap:
        return colmap[col]
    else:
        raise ValueError(r"colname: {0} doesnt map".format(col))
    
def read_ep_data(path=None):
    """
        Read and simple cleaning
    """
    if path is None:
        path = """C:/Users/Brendan/Downloads/ExodusPoint - Assignment Data (Rates Quantitative Strategist) - New York.xlsx"""
    
    xls = pd.ExcelFile(path)
    legacy_sheets = ['TU - Com; NonCom','FV - Com;NonCom', 'TY - Com;NonCom', 'US - Com;NonCom ', 'WN - Com;NonCom']
    tff_sheets = ['TU - Sectorial','FV - Sectorial', 'TY - Sectorial', 'US - Sectorial',  'WN - Sectorial']
    
    a, b = {}, {}
    for s in legacy_sheets:
        d = pd.read_excel(xls, sheet_name=s, index_col=0, header=3, skipfooter=1, parse_dates=True)
        d = d.rename(rename_legacy_cols, axis=1).drop('Total', axis=1)
        ct = s[0:2]
        a[ct] = d
    legacy_raw = pd.concat(a, axis=1)
    
    for s in tff_sheets:
        d = pd.read_excel(xls, sheet_name=s, index_col=0, header=3, skipfooter=1, parse_dates=True)
        d = d.rename(rename_tff_cols, axis=1)
        ct = s[0:2]
        b[ct] = d
    tff_raw = pd.concat(b, axis=1)
    
    futures_raw = pd.read_excel(xls, sheet_name="Futures Price and Duration Data",
                                index_col=0, header=[3,4,5], skipfooter=1, parse_dates=True)
    futures_raw.columns = futures_raw.columns.droplevel(1)
    futures_raw.columns.set_levels([s[0:2] for s in futures_raw.columns.levels[0]], 
                                                         level=0, inplace=True)
    futures_raw.columns.set_names(['ct', 'field'], inplace=True)
    
    swaps_raw = pd.read_excel(xls, sheet_name="Swap Prices",
                                index_col=0, header=3, skipfooter=1, parse_dates=True)
    
    return legacy_raw, tff_raw, futures_raw, swaps_raw


def get_ct_dfs(legacy, tff, futures, swaps):
    """
        Join the data per contract and return a dictionary of dataframes, 
        one for each contract
    """
    legacy = legacy.copy(deep=True)
    tff = tff.copy(deep=True) 
    futures = futures.copy(deep=True)
    swaps = swaps.copy(deep=True)
    cts = futures.columns.levels[0]

    ct_dfs = {}
    for c in cts:
        interm = legacy[c].merge(tff[c], left_index=True, 
                                 right_index=True, how='outer')
        interm = interm.merge(futures[c], left_index=True, right_index=True, how='outer')
        # we dont want the swap closes joined on the data date, we need report date
        interm = interm.merge(swaps[CT_SWAP_MAP[c]], left_index=True, right_index=True, how='outer')
        ct_dfs[c] = interm
    
    return ct_dfs
    
    
def adjust_ct_dfs(ct_dfs, swaps, oi_avg_len=20, adj_rpt=True):
    """
        Compute dv01 and pct average OI as of the data date, 
            but join to swap rates 3-bday hence (approx report date)
    :param adj_rpt: (bool) whether or not to adjust forward to report date
    """
    ret_dfs = {}
    for k, df in ct_dfs.items():
        ct_size = CT_SIZES[k]
        df0 = df.copy(deep=True)
        for c in COT_COLS:
            # num contracts * dur * px/100 * ct_size / 10000
            df0[c+'_dv01']  = df0[c] * df0['FUT_EQV_DUR_NOTL'] * df0['PX_LAST'] * ct_size / 1e6
            df0[c+'_pctAvgOI'] = df0[c] / df0['OPEN_INT'].rolling(oi_avg_len).mean()
        df0 = df0.drop(np.hstack([PRICE_DUR_COLS, CT_SWAP_MAP[k]]), axis=1)
        ret = df0.copy(deep=True)
        if adj_rpt:
            ret = ret.shift(3)
        ret = ret.merge(swaps[CT_SWAP_MAP[k]], left_index=True, 
                      right_index=True, how='outer')
        ret_dfs[k] = ret
    return ret_dfs


def get_rolling_cts(adj_ct_dfs, suffix, window, swap_chg_lags=[1], 
                    swap_chg_fwds=[1], resample_per='W-FRI'):
    """
        Compute data differenced over window periods, resampled on Fridays (release date)
        goal to have stationary series with some memory
        
    :param swap_chg_fwds: (list of ints) days to compute fwd changes in swaps
    """
    ret_dfs = {}
    for k, df in adj_ct_dfs.items():
        df_w = df.copy(deep=True).asfreq(resample_per, method='ffill')
        rel_cols = np.hstack([[c+suffix for c in COT_COLS], CT_SWAP_MAP[k]])
        new_df = df_w.loc[:, rel_cols].copy(deep=True).diff(window)
        first_swap = CT_SWAP_MAP[k][0]        
        # get indicators whether position change corresponds to change in swap yield
# =============================================================================
#         ind_cols = [c+'_ind' for c in COT_COLS]

#         for col, ind in zip([c+suffix for c in COT_COLS], ind_cols):
#             new_df[ind] = (np.sign(new_df[col])!=np.sign(new_df[first_swap])).astype(int)
#             new_df[col+'_rightsign'] = new_df[ind] * new_df[col]
# =============================================================================
        
        # add lagged swap changes
        for i in swap_chg_lags:
            lag = i*window
            new_df[first_swap + '_lag' + str(lag)] = new_df[first_swap].shift(lag)
        for i in swap_chg_fwds:
            fwdname = first_swap + '_fwd' + str(i) 
            new_df[fwdname] = df[first_swap].diff(i).shift(-i)
            new_df[fwdname + '_sign'] = np.sign(new_df[fwdname])
            new_df[fwdname + '_abs'] = np.abs(new_df[fwdname])
        
        ret_dfs[k] = new_df
    
    return ret_dfs


def get_norm_cts(r, swaps, trail_window, swap_chg_fwds=[1]):
    """
        Compute sample standardized features for rolling window changes in positioning
    """
    normed_positioning = {}
    for k, df in r.items():
        ct_df = df.copy(deep=True).dropna(how='all')
        ct_df = ct_df.subtract(ct_df.rolling(trail_window).mean())
        ct_df = ct_df.divide(ct_df.rolling(trail_window).std())
        
        # get par curve levels daily and join fwd chgs
        swap_name = CT_SWAP_MAP[k][0]
        ct_df[swap_name+'_chg1w'] = swaps[swap_name].diff(5)
        for i in swap_chg_fwds:
            fwdname = swap_name + '_fwd' + str(i)
            ct_df[fwdname] = swaps[swap_name].diff(i).shift(-i)
            ct_df[fwdname+'_sign'] = np.sign(ct_df[fwdname])
            ct_df[fwdname+'_abs'] = np.abs(ct_df[fwdname])
            
        normed_positioning[k] = ct_df
        
    return normed_positioning


def get_daily_curve_diffs(curves, r, swaps):
    """
        Get just the daily curve diffs
    :param curves: (list of tuples) combos of curves tuple
    :param r: (dict) of dataframes returned by get_rolling_cts
    :return: dataframe of curve diff 
    """
    curve_diffs = {}
    for curve in curves:
        swap0 = CT_SWAP_MAP[curve[0]][0]
        swap1 = CT_SWAP_MAP[curve[1]][0]
        curve_daily = swaps[swap1] - swaps[swap0]
        curve_diffs["-".join([swap1, swap0])] = curve_daily
    return pd.concat(curve_diffs, axis=1)

    
def get_norm_curve_diffs(curves, r, window, swaps, swap_chg_fwds=[1], norm_window=52):
    """ This computes the differences between ct1 and ct0 of first differences of each feature
         and normalizes the features
    :param curves: (list of tuples) combos of curves tuple
    :param r: (dict) of dataframes returned by get_rolling_cts
    :param window: (int) window over which to compute changes in CoT positions
    :param swaps: (pd.DataFrame) of swap closes
    :param swap_chg_fwds: (list of ints) days to compute fwd changes in swaps
    :return: dict of curve diff dataframes
    """
    curve_diffs = {}
    for curve in curves:
        swap0 = CT_SWAP_MAP[curve[0]][0]
        swap1 = CT_SWAP_MAP[curve[1]][0]

        # first normalize the features
        adj_curve_arr = []
        for c in curve:
            curve_df = r[c].copy(deep=True).dropna(how='all')
            curve_df = curve_df.subtract(curve_df.rolling(norm_window).mean())
            curve_df = curve_df.divide(curve_df.rolling(norm_window).std())
            adj_curve_arr.append(curve_df)


        curve_diff = (adj_curve_arr[1] - adj_curve_arr[0]).dropna(axis=1, how='all')
        
        # compute par curve levels weekly
        swap_curve_name = '-'.join([swap1, swap0])
        curve_diff[swap_curve_name] = r[curve[1]].loc[:, swap1] - r[curve[0]].loc[:, swap0]
        curve_diff[swap_curve_name+'_lag'+str(window)] = curve_diff[swap_curve_name].shift(window)
        
        # compute par curve levels daily then join the fwd chgs
        curve_daily = swaps[swap1] - swaps[swap0]
        for i in swap_chg_fwds:
            fwdname = swap_curve_name+'_fwd'+str(i)
            curve_diff[fwdname] = curve_daily.diff(i).shift(-i)
            curve_diff[fwdname + '_sign'] = np.sign(curve_diff[fwdname])
            curve_diff[fwdname + '_abs'] = np.abs(curve_diff[fwdname])
        
        curve_name = '-'.join(list(reversed(curve)))
        curve_diffs[curve_name] = curve_diff
        
    return curve_diffs

    
def gen_subseq_returns(px, signal, max_len=30):
    """
        Generate ('single contract') returns subsequent to signal firing to evaluate predictiveness
    
    :param px: (pd.Series) series of prices
    :param signal: (pd.Series) series of entry signals (-1,0,1)
    :param max_len: (int) max index lookahead
    
    :return: (pd.DataFrame) returns up to max_len (cols), per signal fire (index)
    """
    signal = signal[signal!=0]
    sub_rets = pd.DataFrame(index=signal.index, columns=np.arange(max_len+1))
    
    for t, sig in signal.iteritems():
        t1_idx = min(px.index.searchsorted(t)+max_len, len(px.index)-1) #end of analysis period
        t1 = px.index[t1_idx]
        px0 = px.loc[t]
        sub_px = px.loc[t:t1].values
        sub_ret_np = np.array(sig*(sub_px-px0))
        if (len(sub_px)<max_len+1):
            sub_ret_np = np.hstack((sub_ret_np,sub_ret_np[-1]*np.ones(max_len+1-len(sub_px))))
        sub_rets.loc[t,:] = sub_ret_np
    
    if (sub_rets.shape[0]!=len(signal)): 
        raise ValueError('Num. of return subsets {0} not equal to num. signals given {1}!'.format(sub_rets.shape[0], 
                                                                                                  len(signal)))
    return sub_rets


def analyze_sub_rets(sub_rets, label, ax=None, subseq_days=[5,20,40], z_threshold=0):
    """
        Generate basic stats summary and plot avg subsequent return series
    
    :param sub_rets: (pd.DataFrame) subsequent returns (columns=days subseq.) per entry signal index
    :return: (pd.Series) summary statistics (and plot avg subsq return series)
    """
    
    avg_sub_rets = sub_rets.mean(axis=0)
    std_sub_rets = sub_rets.std(axis=0)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6), nrows=2, dpi=300)
    label_legend = str(label)+'-wk lookback'
    avg_sub_rets.plot(ax=ax[0],label=label_legend)
    ax[0].set_title('Subsequent Return Avgs: '+str(np.round(z_threshold,2))+'-sigma Threshold')
    ann_ir = avg_sub_rets.divide(std_sub_rets)*np.sqrt(250/sub_rets.columns.values)
    ann_ir.plot(ax=ax[1], label=label_legend)
    ax[1].set_title('Subsequent Return Ann. IR*')
    
    # maybe use framework more in-depth
    stats = ['Mean', 'Std Dev', 'Ann. IR', 'Min','Max', 'Max DD', 'Num Obs', 'Hit Ratio']
    stats_arr = []
    for sp in subseq_days:
        sr = sub_rets.loc[:, sp]
        mean = sr.mean()
        stdev = sr.std()
        ann_ir = mean/stdev*np.sqrt(250/sp)
        mn = sr.min()
        mx = sr.max()
        max_dd = np.min(sub_rets.loc[:, np.arange(sp+1)].values)
        num_obs = len(sr)
        hit_ratio = len(sr[sr>0])/num_obs
        stats_srs = pd.Series([mean, stdev, ann_ir, mn, mx, max_dd, num_obs, hit_ratio], index=stats, name=sp)
        stats_arr.append(stats_srs)
    summary_df = pd.concat(stats_arr, axis=1).round(4)
    return summary_df


def plot_subsq_rets(adj_ct_dfs, ct, feat_adj, filter_feat, windows, swaps, z_threshold, is_start_dt, is_end_dt, 
                    swap_name=None, zscore_lookback=52, max_len=60, curve=False, daily_curves=None):
    """ Plot average returns and annualized IR for returns after a momentum signal trigger """
    if not swap_name:
        swap_name = CT_SWAP_MAP[ct][0]
    if daily_curves is not None:
        srs = daily_curves[swap_name]
    else:
        srs = swaps[swap_name]

    fig, ax = plt.subplots(figsize=(8,8), nrows=2, dpi=500)
    stats_arr = []
    for window in windows:
        r = get_rolling_cts(adj_ct_dfs, feat_adj, window=window, swap_chg_lags=[1], swap_chg_fwds=[1])
        if curve:
            curve_tuple = list(reversed(ct.split('-')))
            normed_pos = get_norm_curve_diffs([curve_tuple], r, window, swaps, norm_window=zscore_lookback)
        else:
            normed_pos = get_norm_cts(r, swaps, zscore_lookback, swap_chg_fwds=[5])
        signals = zscore_signal(normed_pos[ct].loc[is_start_dt:is_end_dt, filter_feat], z_threshold, 'Momentum')
        sub_rets = gen_subseq_returns(srs, signals, max_len=60)
        summary_stats = analyze_sub_rets(sub_rets, label=window, ax=ax, subseq_days=[5,20,60], z_threshold=z_threshold)
        stats_arr.append(summary_stats)
    for a in ax:
        a.legend()
    plt.show()
    stats_df = pd.concat(stats_arr, axis=1, keys=windows, names=['lookback','fwd period'])
    return stats_df


def test_adfuller(input_df, maxlag=1):
    """ return p-vals for unit-root null hyp test """
    tstats = []
    pvals = []
    for name, vals in input_df.iteritems():
        adf = tsa.stattools.adfuller(vals.dropna(), maxlag=maxlag)
        tstats.append(adf[0])
        pvals.append(adf[1])
    results = pd.DataFrame(np.vstack([tstats, pvals]).T, index=input_df.columns, columns=['tstat', 'pval'])
    return results


def _build_regression(endog, exog, model, lasso_positive, alpha):
    """
        Base ridge regression mod builder.
    :param endog: (n x 1 array like) dependent variable
    :param exog: (n x p array like) independent variable(s)
    :param alpha: (float) regularization param
    """
    if model=='Ridge':
        mod = Ridge(alpha=alpha)
    elif model=='Lasso':
        mod = Lasso(alpha=alpha, positive=lasso_positive)
    else:
        raise ValueError("Model must be of type Ridge or Lasso")
    
    mod.fit(endog, exog)
    return mod


def _regression_loop(endog, exog, model, lasso_positive, alpha=50):
    """
        Base regression loop runner, fits ridge and returns insample betas
    :param endog: (n x 1 array like) dependent variable
    :param exog: (n x p array like) independent variable(s)
    :param alpha: (float) regularization param
    """
    mod_result = _build_regression(exog, endog, model, lasso_positive, alpha=alpha)
    beta = np.hstack([mod_result.intercept_, mod_result.coef_])
    
    return beta


def _embargo_ts_splitter(data, test_size, max_train_size=None, embargo=0):
    """
    Scikit-learn time series split with holdout indices
    """
    n_samples = data.shape[0]
    test_embargo_size = test_size+embargo
    n_splits = n_samples//test_embargo_size
    #n_folds = n_splits
    
    
    indices = np.arange(n_samples)
    if n_splits > n_samples:
            raise ValueError(("Cannot have number of folds ={0} greater"
                                 " than the number of samples: {1}.").format(n_splits, n_samples))
    
    test_starts = range(test_embargo_size + n_samples % n_splits, n_samples, test_embargo_size)
    for test_start in test_starts:
        embargo_start = test_start-embargo
        if max_train_size and max_train_size < test_start:
            yield (indices[embargo_start-max_train_size:embargo_start], indices[embargo_start:embargo_start+embargo],
                   indices[test_start:test_start+test_size])
        else:
            yield (indices[:embargo_start], indices[embargo_start:embargo_start+embargo], 
                   indices[test_start:test_start+test_size])
            

def RegressionMain(full_raw, target_col, feature_cols, test_size, model='Ridge', max_train_size=200, embargo_size=1, logpx=True, 
              resample_per='B', ewm_span=50, verbose=False, alpha_override=None, lasso_positive=False):
    """
        Base code to run Ridge framework
    :param full_raw: (pd.DataFrame) of full raw data
    :param target_col: (string) dependent variable col name
    :param feature_cols: (array-like strings) independent variable col name(s)
    :param test_size: (int) out of sample test size
    :param max_train_size: (int) maximum rolling window training size
    :param embargo_size: (int) size of sample to hold out between fits
    :param logpx: (bool) whether to take log of prices to fit
    :param resample_per: (datetimeoffset string) resample period for data
    :param ewm_span: (int) span of EWM to take on data in order to fit
    """
    
    # pre-process
    cols = np.hstack((target_col, feature_cols))
    data = full_raw[cols].copy(deep=True)
    raw_clean = data.asfreq(resample_per).dropna(how='any')
    if ewm_span is not None:
        data = data.ewm(span=ewm_span).mean()
    data = data.asfreq(resample_per).dropna(how='any')
    if logpx:
        data = data.apply(np.log)
        raw_clean = raw_clean.apply(np.log)
        
    dates, betas = [],[]
    
    # get alpha to use in model fits
    ## we only use first quarter of data to not cheat so hard
    x_full = data[feature_cols]
    y_full = data[target_col]
    x_raw_clean = raw_clean[feature_cols]
    
    x_find_alpha = x_full.iloc[:int(data.shape[0]/4)]
    y_find_alpha = y_full.iloc[:int(data.shape[0]/4)]
    tss = TimeSeriesSplit(n_splits=20)
    alpha_space = np.logspace(-6,2,25)
    if model == 'Ridge':
        cv = RidgeCV(alphas=alpha_space, cv=tss)
        cv.fit(x_find_alpha, y_find_alpha)
        alpha = cv.alpha_
        
    elif model == 'Lasso':
        cv = LassoCV(alphas=alpha_space, cv=tss)
        cv.fit(x_find_alpha, y_find_alpha)
        alpha = cv.alpha_
        
    else:
        alpha = 0.0001
    if alpha_override is not None: alpha = alpha_override
    if verbose: print(alpha)
    
    pred_full = pd.Series(name='Pred')
    for train_idx, embargo_idx, test_idx in _embargo_ts_splitter(data, test_size, max_train_size=max_train_size,
                                                                    embargo=embargo_size):
        x_train = x_full.iloc[train_idx]
        y_train = y_full.iloc[train_idx]
        beta = _regression_loop(y_train, x_train, model, lasso_positive, alpha=alpha)
        
        x_test = x_raw_clean.iloc[test_idx]
        pred = sm.add_constant(x_test).dot(beta).rename('Pred')
        pred_full = pred_full.append(pred)
        
        # save to return params
        # date associated with beta (for rebalancing) should be day after computation
        dates.append(data.index[test_idx[0]])
        betas.append(beta)
    
    #rescale if necessary
    if logpx:
        pred_full = pred_full.apply(np.exp)
    
    return pred_full, dates, betas


if __name__=='__main__':
    legacy, tff, futures, swaps = read_ep_data()