# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 22:37:36 2018

@author: Administrator
"""
def clean_list(lst):
    cleaned_list = []
    for item in lst:
        for c in item:
            if c.isalpha() != True:
                item = item.replace(c,'')
        cleaned_list.append(item) #注意append和extend的用法
    return cleaned_list
    
if __name__ == "__main__":
    coffee_list = ['32Latte', '_Americano30', '/34Cappuccino', 'Mocha35']
    cleaned_list = clean_list(coffee_list)
    for i,j in enumerate(cleaned_list):
        print(i+1,j)