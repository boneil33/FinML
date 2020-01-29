# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 15:51:50 2020

@author: Brendan
"""
import os
import pandas as pd
import numpy as np


def fx_data_import(vs_dollar=True, drop=['USDVEF','DTWEXB','DTWEXM','DTWEXO','USDMYR']):
    main_f = "C:/Users/Brendan/MLFin/raw_data/"
    file_names = os.listdir(main_f)
    #japan has the full hist
    file_map = {'DEXBZUS':'USDBRL', 'DEXCAUS':'USDCAD', 'DEXCHUS': 'USDCNY', 'DEXDNUS':'USDDKK',
                'DEXHKUS': 'USDHKD','DEXINUS':'USDINR','DEXJPUS':'USDJPY','DEXKOUS':'USDKRW',
                'DEXMAUS':'USDMYR','DEXMXUS':'USDMXN','DEXNOUS':'USDNOK','DEXSDUS':'USDSEK',
                'DEXSFUS':'USDZAR','DEXSLUS':'USDSLR','DEXSZUS':'USDCHF','DEXSIUS':'USDSGD',
                'DEXTAUS':'USDTWD','DEXTHUS':'USDTHB','DEXUSAL':'AUDUSD','DEXUSEU':'EURUSD',
                'DEXUSNZ':'NZDUSD','DEXUSUK':'GBPUSD','DEXVZUS':'USDVEF'
     }
    frames = []
    for f in file_names:
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