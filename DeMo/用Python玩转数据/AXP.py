# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 22:07:59 2018

@author: Administrator
"""

import requests,re,json
import pandas as pd
from datetime import date

def retrieve_quotes_historical(stock_code):
    quotes = []
    url = 'https://finance.yahoo.com/quote/%s/history?p=%s'%(stock_code,stock_code)
    r =requests.get(url)
    m = re.findall('"HistoricalPriceStore":{"prices":(.*?),"isPending"',r.text)
    if m:
        quotes = json.loads(m[0])
        quotes = quotes[::-1]
    return [item for item in quotes if not 'type' in item]

quotes = retrieve_quotes_historical('AXP')
quotesdf_ori = pd.DataFrame(quotes)
quotesdf = quotesdf_ori.drop(['adjclose'],axis=1)

quotes2 = retrieve_quotes_historical('IBM')
list = []
for i in range(len(quotes2)):
    x= date.fromtimestamp(quotes2[i]['date'])
    y= date.strftime(x,'%Y-%m-%d')
    list.append(y)
quotesdf2_ori = pd.DataFrame(quotes2,index=list)
quotesdf2 = quotesdf2_ori.drop(['date'],axis=1)



#quotesdf = pd.read_csv(r'd:\Anaconda3\DeMo\用Python玩转数据\AXP.csv')