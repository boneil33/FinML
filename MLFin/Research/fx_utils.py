# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 15:51:50 2020

@author: Brendan
"""
import os
import pandas as pd
import numpy as np


def fx_data_import(vs_dollar=True, drop=[]):
    main_f = r"C:/Users/Brendan/FinML/MLFin/raw_data/"
    file_names = os.listdir(main_f)
    #japan has the full hist
    file_map = {'DEXBZUS':'USDBRL', 'DEXCAUS':'USDCAD', 'DEXCHUS': 'USDCNY', 'DEXDNUS':'USDDKK',
                'DEXHKUS': 'USDHKD','DEXINUS':'USDINR','DEXJPUS':'USDJPY','DEXKOUS':'USDKRW',
                'DEXMXUS':'USDMXN','DEXNOUS':'USDNOK','DEXSDUS':'USDSEK',
                'DEXSFUS':'USDZAR','DEXSLUS':'USDSLR','DEXSZUS':'USDCHF','DEXSIUS':'USDSGD',
                'DEXTAUS':'USDTWD','DEXTHUS':'USDTHB','DEXUSAL':'AUDUSD','DEXUSEU':'EURUSD',
                'DEXUSNZ':'NZDUSD','DEXUSUK':'GBPUSD'
     }
    frames = []
    for f in file_map.keys():
        f+='.csv'
        df = pd.read_csv(main_f+f, index_col=0, parse_dates=True)
        df['VALUE'] = pd.to_numeric(df['VALUE'], 
          errors='coerce').fillna(method='ffill')
        df = df.rename({'VALUE': f[:-4]}, axis=1)
        frames.append(df)
    
    full_data = pd.concat(frames, axis=1, join='outer', 
                          verify_integrity=True).fillna(method='ffill')
    
    full_data = full_data.rename(file_map, axis=1)
    
    if vs_dollar:
        # price everything in USD terms
        for name,s in full_data.iteritems():
            if name[:3] != 'USD' and name[3:] == 'USD':
                full_data['USD'+name[:3]] = 1./s
                full_data.drop(name, axis=1, inplace=True)
    full_data=full_data.drop(drop, axis=1)
    return full_data


def bbg_data_import(vs_dollar=True):
    close = pd.read_excel('V:/data/fx_data.xlsx', sheet_name='fx_data', index_col=0, na_values='ND')
    yields = pd.read_excel('V:/data/fx_data.xlsx', sheet_name='Yields', index_col=0, skiprows=2)
    close = close.dropna(how='all')
    close = close.fillna(method='ffill')
    
    if vs_dollar:
        # price everything in USD terms
        for name,s in close.iteritems():
            
            if name[:3] != 'USD' and name[3:] == 'USD':
                close['USD'+name[:3]] = 1./s
                close.drop(name, axis=1, inplace=True)
        for name, s in yields.iteritems():
            if name[:3] != 'USD' and name[3:] == 'USD':
                yields['USD'+name[:3]] = 1./s
                yields.drop(name, axis=1, inplace=True)
    
    return close, yields


if __name__=='__main__':
    bbg_data_import()
