# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 07:23:51 2020

@author: Hudson-and-Thames/mlfinlab
very lightly adjusted by Brendan
"""

import pandas as pd
import numpy as np
import datetime as dt


class ETFTrick():
    """
    "ETF Trick" class for creating synthetic pnl series of a set of products
    """
    
    def __init__(self, open_df, close_df, alloc_df, carry_df, rates_df=None, rebal_freq='W-FRI'):
        """
        Constructor
        
        Creates class object
        :param open_df: (pd.DataFrame) of open prices
        :param close_df: (pd.DataFrame) of close prices
        :param alloc_df: (pd.DataFrame) of portfolio weights
            note if this never changes, pnl does not get reinvested,
            which can create negative AUM
        :param carry_df: (pd.DataFrame) of carry in units of underlying price
        :param rates_df: (pd.DataFrame) of dollar value of one point in 
            the underlying contract, includes fx and futures multipliers
        :param rebal_freq: (string) rebalance frequency if weights don't change
        """
        
        # if last index we rebalance then price change will be open to close
        self.prev_allocs_change = False
        self.rebal_freq = rebal_freq
        self.prev_h = None # to get current value we need last holdings array
        self.data_dict = {}
        
        self.data_dict['open'] = open_df
        self.data_dict['close'] = close_df
        self.data_dict['alloc'] = alloc_df
        self.data_dict['carry'] = carry_df
        self.data_dict['rates'] = rates_df
        self.securities = self.data_dict['alloc'].columns
        
        if rates_df is None:
            self.data_dict['rates'] = open_df.copy()
            self.data_dict['rates'][self.securities] = 1.0
        
        #align
        for name in self.data_dict:
            self.data_dict[name] = self.data_dict[name][self.securities]
        
        # check indices, initialize ETF value to 1
        self._index_check()
        self.prev_k = 1.0
        # initialize portfolio weights
        self.prev_allocs = alloc_df.iloc[0]
        
        
    
    def _index_check(self):
        """
        Internal check that all incides are aligned
        """
        for temp in self.data_dict.values():
            if (self.data_dict['open'].index.difference(temp.index).shape[0]
                != 0) or (self.data_dict['open'].shape != temp.shape):
                
                raise ValueError('Dataframe indices are not aligned')
        
    
    def _generate_trick_components(self, output=False):
        """
        Calculates all vectorizable data needed
        Components:
            w: alloc df
            h: h/K holdings of each security / K since K is iterative
            close_open: open to close price change
            close_diff: close price change
            carry: carry_df
            rate: rates_df
        
        :return: (pd.DataFrame) (num_components*num_securities) columns
            component_1:asset_1, component_2:asset_1, ..., component_6:asset_n
        """
        
        close_diff = self.data_dict['close'].diff()
        next_open = self.data_dict['open'].shift(-1) # next open px
        close_open = self.data_dict['close'].subtract(self.data_dict['open'])
        # total weights
        self.data_dict['alloc']['abs_w_sum'] = self.data_dict['alloc'].abs().sum(axis=1)
        
        # delevered allocations
        delever = self.data_dict['alloc'].divide(self.data_dict['alloc']['abs_w_sum'], axis=0)
        next_open_dollar = next_open.multiply(self.data_dict['rates'], axis=0)
        
        # get indices at which you force rebal
        data_index = self.data_dict['open'].index
        if self.rebal_freq is not None:
            rebal_index = self.data_dict['open'].asfreq(self.rebal_freq).dropna(how='any').index
        else:
            rebal_index = []
        reset_index = [idx in rebal_index for idx in data_index]
        reset_df = pd.Series(reset_index, index=data_index, name='force_rebal')
        
        # get holdings vector without pnl
        h_without_k = delever.divide(next_open_dollar)
        weights = self.data_dict['alloc'][self.securities]
        
        # compute abs change in weights (per $ total ETF) for transaction cost analysis
        weights_diff_delev = weights.diff(1).fillna(0).divide(self.data_dict['alloc']['abs_w_sum'], 
                                                              axis=0).abs().sum(axis=1)
        
        h_without_k = h_without_k[self.securities]
        close_open = close_open[self.securities]
        close_diff = close_diff[self.securities]
        
        # everything together
        final = pd.concat([weights, weights_diff_delev, h_without_k, close_open, close_diff,
                           self.data_dict['carry'], self.data_dict['rates'],
                           reset_df], axis=1,
                           keys=['weights', 'weights_diff_delev', 'holdings','close_open',
                                 'close_diff','carry','rates','force_rebal'])
        if output:
            final.to_excel('/home/boneil/data/trs_output_components.xlsx')
        return final
    
    
    def _chunk_loop(self, data_df):
        """
        ETF Trick iteration looper to calculate trick value series
        
        :param data_df: (pd.DataFrame) set to apply trick on, columns
            multiindex will be:
                level 0 : 'w','h','close_open','close_diff','carry','rates'
        :return: (pd.Series) ETF Trick time series values
        """
        etf_series = pd.Series()
        
        # trying more pandastic way than hudson and thames
        for index, row in data_df.iterrows():
            row = row.fillna(0)
            weights_change = bool(~(self.prev_allocs == 
                                    row['weights']).all())
            if self.prev_allocs_change:
                delta = row['close_open']
            else:
                delta = row['close_diff']
            
            if self.prev_h is None:
                # previous h is needed to calculate current value of ETF
                self.prev_h = row['holdings']*self.prev_k
                # K = 1 to start!
                etf_series.loc[index] = self.prev_k
            else:
                # if rebalanced, holdings need to include pnl
                # otherwise, just equal to previous
                if self.prev_allocs_change:
                    self.prev_h = row['holdings'] * self.prev_k
                k = self.prev_k + (row['rates']*(delta+row['carry']) *
                                   self.prev_h).sum()
                etf_series[index] = k
                self.prev_k = k
                # update prev allocs
                self.prev_allocs_change = (weights_change or 
                                           data_df[('force_rebal',
                                                    'force_rebal')].loc[index])
                self.prev_allocs = row['weights']

        return etf_series
     
                
    def get_etf_series(self, output_inter=False, return_data=False):
        """
        External method to retrieve ETF series
        :return: (pd.Series) time series of synthetic ETF values
        """
        data = self._generate_trick_components(output=output_inter)
        
        # delete first row that will have NaNs from price diffs
        #data = data
        etf = self._chunk_loop(data)
        if return_data:
            return etf, data
        return etf
    
