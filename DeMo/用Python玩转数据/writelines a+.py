# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:32:15 2018

@author: Administrator
"""
s = 'I like bananas'
with open('D:\\Anaconda3\\DeMo\\firstpro.txt','a+') as f:
    f.seek(0)
    f.writelines('\n')
    f.writelines(s)
    f.seek(0)
    a = f.readlines()
    print(a)


