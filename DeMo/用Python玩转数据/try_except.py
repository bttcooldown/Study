# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:32:15 2018

@author: Administrator
"""

while True:
    try:
        count = int(input("Enter count: "))
        price = int(input("Enter price for each one: "))
        Pay = count * price
        print("The price is: ", Pay)
        break
    except:
        print('Error, please enter numeric one. ')