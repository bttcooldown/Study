# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')



train = pd.read_csv(r'D:\Python files\DeMo\House Prices\train.csv')
train_Id = train['Id']
train.drop('Id',axis=1,inplace=True)

#热图
corrmat = train.corr()
sns.heatmap(corrmat,vmax=0.8,square=True)

cols = corrmat.nlargest(10,'SalePrice').index
cm = np.corrcoef(train[cols].T)
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

#与SalePrice相关性较高的有 OverallQual, GrLivArea,GarageCars, GargeArea, TotalBsmtSF, 1stFlrSF, FullBath, TotRmsAbvGrd, YearBuilt, YearRemodAdd, GargeYrBlt, MasVnrArea and Fireplaces. 
#- 这其中，GarageArea和GarageCars,GarageYrBlt有很强的相关性，可以选择有代表性的GarageCars; 
#- TotalBsmtSF和1stFlrSF有很强相关性，可以作为新特征Basement分析； 
#- TotRmsAbvGrd与GrLivArea相似，取GrLivArea分析。

#直方图 偏度和峰度
sns.distplot(train.SalePrice)
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())

#散点图
train[['SalePrice','GrLivArea']].plot.scatter(x='GrLivArea',y='SalePrice', ylim=(0,800000))

#用seaborn的.pairplot() 画很多散点图
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols],size=2.5)
plt.show()

#盒图
fig = sns.boxplot(train.OverallQual, train.SalePrice)
fig.axis(ymin=0, ymax=800000)


#得到特征属性（数值和类别）
num_f = [f for f in train.columns if train.dtypes[f] != 'object']
num_f.pop()#去掉SalePrice
category_f = [f for f in train.columns if train.dtypes[f] == 'object']

#缺失值
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1,keys=['Total', 'Percent'])
missing_data.head(20)


for num in num_f:
    train[num].fillna(0,inplace=True)
for category in category_f:
    train[category].fillna('None',inplace=True)
train['LotFrontage'] = train.groupby(['LotArea','Neighborhood'])['LotFrontage'].transform(lambda x:x.fillna(x.median()))   

#特征工程
train.groupby(['MSSubClass'])['SalePrice'].agg(['mean','median','count'])

#离群点 单变量分析 可以发现，Low range值偏离原点并且都比较相近，High range离远点较远，7.很可能是异常值
saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'].values.reshape(-1,1))
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

#双变量分析 （根据'SalePrice','GrLivArea'的 散点图）
train.sort_values(by = 'GrLivArea', ascending = False)[:2]
train.drop(train[(train['GrLivArea']>4000)&(train.SalePrice<300000)].index,inplace=True)

#正态化
sns.distplot(train['SalePrice'], fit=norm)
plt.figure()
stats.probplot(train['SalePrice'], plot=plt)

sns.distplot(np.log(train['SalePrice']), fit=norm)
plt.figure()
stats.probplot(np.log(train['SalePrice']), plot=plt)





