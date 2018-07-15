# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:32:15 2018

@author: Administrator
"""
import numpy as np
import time,math
x = np.arange(0,100,0.01)
t_m1 = time.clock()
for i,t in enumerate(x):
    x[i] = math.pow((math.sin(t)),2)
t_m2 = time.clock()
y = np.arange(0,100,0.01)
t_n1 = time.clock()
y = np.power(np.sin(y),2)
t_n2 = time.clock()
print('Running time of math:',t_m2-t_m1)
print('Running time of numpy:',t_n2-t_n1)

