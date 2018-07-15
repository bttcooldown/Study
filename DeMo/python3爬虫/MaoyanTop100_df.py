# -*- coding: utf-8 -*-
"""
Created on Fri May 18 23:25:03 2018

@author: Administrator
"""
import pandas as pd


a=b=[]
with open('result.txt','r',encoding='utf-8') as f:
    a=f.readlines()
for i in range(100):
    a[i]=a[i][0:-1] #去掉换行符
    b.append(eval(a[i])) #str转换dict
df = pd.DataFrame(b)