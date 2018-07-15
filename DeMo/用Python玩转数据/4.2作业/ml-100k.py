# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 20:06:26 2018

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import time

rating = pd.read_table('u.data', sep='\t', names=None)
rating.columns = ['user id','item id','rating','timestamp']
userinfo = pd.read_table('u.user', sep='|', names=None)
userinfo.columns = ['user id','age','gender','occupation','zip code']
rating2 = rating[['user id','rating']]
userinfo2 = userinfo[['user id','gender']]
rating_gender = pd.merge(rating2,userinfo2)
rating_gender_M = rating_gender[rating_gender['gender']=='M']
rating_gender_F = rating_gender[rating_gender['gender']=='F']
mean_M = rating_gender_M.groupby('user id').rating.mean()
mean_F = rating_gender_F.groupby('user id').rating.mean()
std_M = mean_M.std()
std_F = mean_F.std()