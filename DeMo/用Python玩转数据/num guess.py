# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 21:43:05 2018

@author: Administrator
"""

from random import randint
x = randint(0,300)
go = 'y'
while (go == 'y'):
    digit = int(input('Pleae input a number between 0~300'))
    if digit == x:
        print('Bingo!')
        break
    elif digit > x:
        print('Too large,please try again')
    else:
        print('Too small,please try again')
    print('Input y if you want to continue.')
    go = input()
    print(go)
else:
    print('Goodbye!')