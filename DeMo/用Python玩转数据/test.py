# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 19:55:57 2018

@author: Administrator
"""

def find_person(dict_users, strU):
      if strU in dict_users.keys():
          return dict_users[strU]
      else:
          return 'Not Found'

if __name__ == "__main__":
     dict_users = {'xiaoyun':88888,'xiaohong':5555555,'xiaoteng':11111,'xiaoyi':12341234,'xiaoyang':1212121}
     strU =  input('Please input the name: ')
     print(find_person(dict_users, strU))