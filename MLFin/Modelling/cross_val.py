# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:07:56 2020

@author: Brendan
"""

import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import _BaseKFold


class PurgedKFold(_BaseKFold):
    """
    Extend scikit-learn's KFold cross val class to work with trading strategies
    In particular, 'purge' training sets of observations that overlap with
        information used in the test set
    Additionally, 'embargo' additional train data post-test set to help address
        serial correlation
    """
    def __init__(self, n_splits, t1=None, pct_embargo=0.):
        if not isinstance(t1, pd.Series):
            raise ValueError('label start/end dates must be pandas Series)
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pct_embargo=pct_embargo
    
    def split(self, X, y=None, groups=None):
        if (X.index==self.t1.index).sum()!=t1.shape:
            raise ValueError('features matrix must completely cover t1 start dates')
        indices = np.arange(X.shape[0])
        embargo = int(X.shape[0]*self.pct_embargo)
        test_starts = [(i[0],i[-1]+1) for i in 
                       np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for i,j in test_starts:
            t0 = self.t1.index[i] # test start date
            test_indices = indices[i:j]
            max_t1_idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            # train indices where signal end date prior to test start
            train_indices_1 = self.t1.index.searchsorted(self.t1[self.t1<=t0].index)
            # append indices after purge and embargo
            train_indices = np.concatenate((train_indices_1, indices[max_t1_idx+embargo:]))
            yield train_indices, test_indices
            