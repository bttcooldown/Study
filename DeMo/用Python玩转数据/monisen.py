# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 22:02:29 2018

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

def monisen():
    n=4
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

monisen()
print(M)