# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 21:06:01 2018

@author: Administrator
"""


def clean_data(df, drop_passenger_id):
    import pandas as pd
    import numpy as np
    # 性别种类
        sexes = sorted(df['Sex'].unique())
    
    # 生成从字符串到数字表示的性别映射    
    genders_mapping = dict(zip(sexes, range(0, len(sexes) + 1)))

    # 将性别从字符串转换为数字表示
    df['Sex_Val'] = df['Sex'].map(genders_mapping).astype(int)
    
    # 登船位置种类
    embarked_locs = sorted(df['Embarked'].fillna('ANN').unique())

    # 生成从字符串到数字表示登船的开始映射        
    embarked_locs_mapping = dict(zip(embarked_locs, 
                                     range(0, len(embarked_locs) + 1)))

    df['Embarked'] = df['Embarked'].fillna('ANN')
    df['Embarked_Val'] = df['Embarked'].map(embarked_locs_mapping).astype(int)
    # 填入缺失的登船信息
    # 因为绝大多数乘客都是乘坐“S”号登船的 
    # 我们将丢失的值是登船的'S'种类
    if len(df[df['Embarked']=='ANN'] > 0):
        df.replace({'Embarked_Val' : 
                       { embarked_locs_mapping['ANN'] : embarked_locs_mapping['S'] 
                       }
                   }, 
                   inplace=True)
    
    # 转换从一个字符串开始到虚拟变量    
    df = pd.concat([df, pd.get_dummies(df['Embarked_Val'], prefix='Embarked_Val')], axis=1)
    
    # 以平均票价补充缺失的票价
    if len(df[df['Fare'].isnull()] > 0):
        avg_fare = df['Fare'].mean()
        df.replace({ None: avg_fare }, inplace=True)
    
    # 为保证年龄列不变, 为了保持年龄的增长，制作一个叫做AgeFill的副本。 
    df['AgeFill'] = df['Age']

    # 通过Sex_Val确定每个乘客类别的年龄.  
    # 我们用中位数而不是平均值，因为年龄直方图似乎是右倾斜的 
    df['AgeFill'] = df['AgeFill'] \
                        .groupby([df['Sex_Val'], df['Pclass']]) \
                        .apply(lambda x: x.fillna(x.median()))
            
    #定义一个新特性FamilySize他是Parch(父母或孩子的数量)和SibSp(兄弟姐妹或配偶的数量) 
    df['FamilySize'] = df['SibSp'] + df['Parch']
    
    # 删除我们不使用的列
    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
    
    # 删除Age列，因为我们将使用AgeFill列代替
    # 放弃SibSp和Parch列，因为我们将使用FamilySize
    # 删除乘客id列，因为它不会被用作特性
    df = df.drop(['Age', 'SibSp', 'Parch'], axis=1)
    
    if drop_passenger_id:
        df = df.drop(['PassengerId'], axis=1)
    
    return df