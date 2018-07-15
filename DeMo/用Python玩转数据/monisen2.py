# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:32:15 2018

@author: Administrator
"""
from math import sqrt
def prime(num):
    if num == 1:
        return False
    k = int(sqrt(num))
    for i in range(2,k+1):
        if num % i == 0:
            return False
            break
    return True

n=6
j=0
i=2
while (j != n):
    if prime(i):
        M = 2**i-1
        if prime(M):
            j += 1
            i += 1
            continue
    i += 1
print(M)