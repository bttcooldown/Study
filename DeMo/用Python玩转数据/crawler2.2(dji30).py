# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 19:52:52 2018

@author: Administrator
"""

import requests,re
import pandas as pd

def retrieve_dji_list():
    r = requests.get('http://money.cnn.com/data/dow30/')
    search_pattern = re.compile('class="wsod_symbol">(.*?)<\/a>.*?<span.*?">(.*?)<\/span>.*?\n.*?class="wsod_stream">(.*?)<\/span>')
    dji_list_in_text = re.findall(search_pattern,r.text)
    dji_list = []
    for item in dji_list_in_text:
        dji_list.append([item[0],item[1],float(item[2])])
    return dji_list

dji_list = retrieve_dji_list()
djidf = pd.DataFrame(dji_list)
djidf.index = range(1,31)
djidf.columns = list(['code','names','lasttrade'])
#print(djidf)