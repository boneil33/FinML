# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 13:14:47 2020

@author: Brendan
"""

import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dt
import itertools
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from Preprocessing.labeling import resample_close
from Research.fx_utils import fx_data_import
from Modelling.cross_val import PurgedKFold
from sklearn.metrics import log_loss, accuracy_score


def get_pca_weights(close, window, n_components=None, ret_pca=False):
    """
    Get rolling PCA weights
    
    :param close: (pd.DataFrame) close prices, sampled how you like
    :param window: (int) length of window used to compute pca
    :param n_components: (int) number of components to return weights on
    :return: (pd.DataFrame) PC weights
        index = timeseries name
        cols = PC
    """
    rets = close.apply(np.log).diff().dropna()
    rets_centered = rets.subtract(rets.rolling(window).mean()).dropna()
    vols = rets.rolling(window).std().dropna()
    std_rets = rets_centered.divide(vols).dropna()
    std_rets_sub = std_rets.iloc[-window:]
    
    pca = PCA(n_components=n_components)
    try:
        pca.fit(std_rets_sub)
    except ValueError as v:
        print(v)
        print(std_rets.tail())
        print(std_rets_sub.head())
        raise v
    components_df = pd.DataFrame(pca.components_, index=np.arange(pca.n_components_),
                                 columns=close.columns).T
    
    if ret_pca:
        return components_df, pca
    return components_df


def get_cluster_labels(components_df, random_state=None, n_clusters=5, components_to_use=None):
    """
    Get KMeans cluster labels
    
    :param components_df: (pd.DataFrame) component weights 0:n_components cols
    :param random_state: (int) random state seed to be consistent across calls
    :param n_clusters: (int) number of clusters
    :return: (pd.Series) cluster labels
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    if components_to_use is None:
        kmeans.fit(components_df)
    else:
        kmeans.fit(components_df[components_to_use])
    #labels = pd.Series(kmeans.labels_, index=components_df.index)
    return kmeans.labels_


def get_pca_distances(components_df):
    """
    Get pairwise component distances in PC space between various series
    
    :param components_df: (pd.DataFrame) component weights 0:n_components cols
    :return: (pd.DataFrame) matrix of pairwise component distances on pc space
    """
    d = pd.DataFrame()
    for name_i, srs_i in components_df.T.iteritems():
        
        for name_j, srs_j in components_df.T.iteritems():
            d.loc[name_i,name_j] = np.linalg.norm(srs_i-srs_j)
    return d       
    

def pca_cluster_loop(close, window, n_components, n_clusters, components_to_use=None):
    """
    Main loop to generate timeseries of security return clusters from pca weights
    
    :param close: (pd.DataFrame) close prices
    :param window: (int) length of window used to compute pca
    :param n_components: (int) number of components to return weights on
    :param n_clusters: (int) number of clusters
    :return: (pd.DataFrame) timeseries of cluster labels per series
        
    """
    rand_state = np.random.randint(0,100)
    close = close.dropna(how='any')
    cluster_ts = pd.DataFrame(np.zeros(close.shape), index=close.index, columns=close.columns)

    # idx will be the index at which the window ends
    # idx-window wlil be the index at which the window starts
    # idx-window*2:idx-window is where we use some data to compute rolling mean+std
    sample_window = window*2
    for idx in np.arange(sample_window,close.shape[0],window):
        sub_close = close.iloc[idx-sample_window:idx]
       
        components = get_pca_weights(sub_close, window, n_components=n_components)
        labels = get_cluster_labels(components, random_state=rand_state, n_clusters=n_clusters,
                                    components_to_use=components_to_use)
        for i in range(window):
            if idx+i == cluster_ts.shape[0]:
                break
            cluster_ts.iloc[idx+i] = labels
        
    return cluster_ts
    

def aux_feat_imp_sfi(features, clf, X, events, scoring='neg_log_loss', n_splits=5, pct_embargo=0.01):
    """
    Compute single feature importance
    """
    imp = pd.DataFrame(columns=['mean','std'])
    p_kfold = PurgedKFold(n_splits=n_splits, t1=events['t1'], pct_embargo=0.01)
    for name in features:
        df0 = p_kfold.cv_score(clf, X=X.loc[:,name].to_frame(), y=events['bin'], 
                               sample_weight=events['w'], scoring=scoring)
        imp.loc[name, 'mean'] = df0.mean()
        imp.loc[name, 'std'] = df0.std()*(df0.shape[0]**-0.5)
    
    return imp


def feat_imp_mda(clf, X, events, scoring='neg_log_loss', n_splits=5, pct_embargo=0.01):
    """
    Compute mean decrease accuracy (OoS)
    """
    if scoring not in ['neg_log_loss','accuracy']:
        raise ValueError('only implemented for neg_log_loss and accuracy, got: ', scoring)
    y = events['bin']
    p_kfold = PurgedKFold(n_splits=n_splits, t1=events['t1'], pct_embargo=0.01)
    score0 = pd.Series()
    score1 = pd.DataFrame(columns=X.columns)
    for i, (train,test) in enumerate(p_kfold.split(X=X)):
        x0,y0,w0 = X.iloc[train,:], y.iloc[train], events['w'].iloc[train]
        x1,y1,w1 = X.iloc[test,:], y.iloc[test], events['w'].iloc[test]
        fit = clf.fit(X=x0, y=y0, sample_weight=w0.values)
        if scoring=='neg_log_loss':
            prob = fit.predict_proba(x1)
            score0.loc[i] = -log_loss(y1, prob, sample_weight=w1.values,
                                      labels=clf.classes_)
        else:
            pred = fit.predict(x1)
            score0.loc[i] = accuracy_score(y1, pred, sample_weight=w1.values)
        
        # permute data and refit
        for col in X.columns:
            x1_ = x1.copy(deep=True)
            np.random.shuffle(x1_[col].values)
            if scoring=='neg_log_loss':
                prob = fit.predict_proba(x1_)
                score1.loc[i, col] = -log_loss(y1, prob, sample_weight=w1.values,
                                             labels=clf.classes_)
            else:
                pred = fit.predict(x1_)
                score1.loc[i, col] = accuracy_score(y1, pred, sample_weight=w1.values)
        
    # compare permuted data fit to 'true' fit for each feature
    imp = (-score1).add(score0, axis=0)
    if scoring=='neg_log_loss':
        imp = imp/-score1
    else:
        imp = imp/(1.-score1)
    
    imp = pd.DataFrame({'mean':imp.mean(), 'std':imp.std()*imp.shape[0]**-0.5})
    return imp, score0.mean()
    

if __name__=='__main__':
    n_comp = 3
    close = fx_data_import(vs_dollar=True).drop(['USDVEF','DTWEXB','DTWEXO','DTWEXM'],axis=1)
    resampled = resample_close(close, period='W-FRI')
    clusters = pca_cluster_loop(resampled, 100, 3, 5, components_to_use=[1,2])
    components = get_pca_weights(close, 100, n_components=5)
    d=get_pca_space_distances(components)
    
