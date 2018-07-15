# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:32:15 2018

@author: Administrator
"""
with open('D:\\Anaconda3\\DeMo\\firstpro.txt') as f:
    a = f.readlines()
    for i in range(0,len(a)):
        a[i] = str(i+1) + ' ' + a[i]
with open('D:\\Anaconda3\\DeMo\\firstpro2.txt','w') as f2:
    f2.writelines(a)

