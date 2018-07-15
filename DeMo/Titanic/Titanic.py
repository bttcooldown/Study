# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 22:47:17 2018

@author: Administrator
"""

import pandas as pd
import numpy as np
import pylab as plt

# Set the global default size of matplotlib figures
plt.rc('figure', figsize=(10, 5))

# Size of matplotlib figures that contain subplots
fizsize_with_subplots = (10, 10)

# Size of matplotlib histogram bins
bin_size = 10

df_train = pd.read_csv('train.csv')

fig = plt.figure(figsize=fizsize_with_subplots)
fig_dims = (3,2)
# Plot death and survival counts
plt.subplot2grid(fig_dims,(0,0))
df_train['Survived'].value_counts().plot(kind='bar',title='Death and Survival Counts')

# Plot Pclass counts
plt.subplot2grid(fig_dims, (0, 1))
df_train['Pclass'].value_counts().plot(kind='bar', 
                                       title='Passenger Class Counts')

# Plot Sex counts
plt.subplot2grid(fig_dims, (1, 0))
df_train['Sex'].value_counts().plot(kind='bar', 
                                    title='Gender Counts')
plt.xticks(rotation=0)

# Plot Embarked counts
plt.subplot2grid(fig_dims, (1, 1))
df_train['Embarked'].value_counts().plot(kind='bar', 
                                         title='Ports of Embarkation Counts')

# Plot the Age histogram
plt.subplot2grid(fig_dims, (2, 0))
df_train['Age'].hist()
plt.title('Age Histogram')

#1.Feature: Passenger Classes
pclass = pd.crosstab(df_train['Pclass'], df_train['Survived'])
pclass_pct=pclass.div(pclass.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True)

#2.Feature: Sex
sexes = sorted(df_train['Sex'].unique())
genders_map = dict(zip(sexes,range(0,len(sexes)+1)))
df_train['Sex_val'] = df_train['Sex'].map(genders_map).astype(int)
sex_survive = pd.crosstab(df_train['Sex_val'],df_train['Survived'])
sex_survive_pic = sex_survive.div(sex_survive.sum(1).astype(float),axis=0)
sex_survive_pic.plot(kind='bar',stacked=True,title='Survival Rate by Gender')

pass_classes = sorted(df_train['Pclass'].unique())
for p_class in pass_classes:
    print('M',p_class,len(df_train[(df_train['Sex']=='male')&(df_train['Pclass']==p_class)]))
    print('F',p_class,len(df_train[(df_train['Sex']=='female')&(df_train['Pclass']==p_class)]))
    
female_df = df_train[df_train['Sex']=='female']
female_survive = pd.crosstab(female_df['Pclass'],female_df['Survived'])
female_survive_pic = female_survive.div(female_survive.sum(1).astype(float),axis=0)
female_survive_pic.plot(kind='bar',stacked=True,title='Female Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')

male_df = df_train[df_train['Sex']=='male']
male_survive = pd.crosstab(male_df['Pclass'],male_df['Survived'])
male_survive_pic = male_survive.div(male_survive.sum(1).astype(float),axis=0)
male_survive_pic.plot(kind='bar',stacked=True,title='Male Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')

#3.Feature: Embarked
df_train[df_train['Embarked'].isnull()]
embarked = sorted(df_train['Embarked'].fillna('ANN').unique())
embarked_map = dict(zip(embarked,range(0,len(embarked)+1)))
df_train['Embarked'] = df_train['Embarked'].fillna('ANN')  #只能先转换NaN，否则无法和str排序
df_train['Embarked_val'] = df_train['Embarked'].map(embarked_map).astype(int)

df_train['Embarked_val'].hist(bins=len(embarked), range=(0, 3))
plt.title('Port of Embarkation Histogram')
plt.xlabel('Port of Embarkation')
plt.ylabel('Count')
plt.show()

df_train.replace({'Embarked_val' : 
                   { embarked_map['ANN'] :embarked_map['S'] 
                   }
               }, 
               inplace=True)
embarked = sorted(df_train['Embarked_val'].unique())

embarked_survive = pd.crosstab(df_train['Embarked_val'], df_train['Survived'])
embarked_survive_pic = embarked_survive.div(embarked_survive.sum(1).astype(float),axis=0)
embarked_survive_pic.plot(kind='bar', stacked=True)
plt.title('Survival Rate by Port of Embarkation')
plt.xlabel('Port of Embarkation')
plt.ylabel('Survival Rate')

#生成Embarked_val的dummy variables
df_train = pd.concat([df_train, pd.get_dummies(df_train['Embarked_val'], prefix='Embarked_val')], axis=1)

#4.Feature: Age
df_train['AgeFill'] = df_train['Age']
df_train['AgeFill'] = df_train['AgeFill'] \
                        .groupby([df_train['Sex_val'], df_train['Pclass']]) \
                        .apply(lambda x: x.fillna(x.median()))
fig,axes=plt.subplots(2, 1, figsize=fizsize_with_subplots)

# Histogram of AgeFill segmented by Survived
df1 = df_train[df_train['Survived']==0]['Age']
df2 = df_train[df_train['Survived']==1]['Age']
max_age = max(df_train['AgeFill'])
axes[0].hist([df1, df2], 
             bins=max_age / bin_size, 
             range=(1, max_age), 
             stacked=True)
axes[0].legend(('Died', 'Survived'), loc='best')
axes[0].set_title('Survivors by Age Groups Histogram')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Count')

# Scatter plot Survived and AgeFill
axes[1].scatter(df_train['Survived'], df_train['AgeFill'])
axes[1].set_title('Survivors by Age Plot')
axes[1].set_xlabel('Survived')
axes[1].set_ylabel('Age')

#Plot AgeFill density by Pclass:
for pclass in pass_classes:
    df_train['AgeFill'][df_train['Pclass']==pclass].plot(kind='kde')
plt.title('Age Density Plot by Passenger Class')
plt.xlabel('Age')
plt.legend(('1st Class', '2nd Class', '3rd Class'), loc='best')    

# Set up a grid of plots
fig = plt.figure(figsize=fizsize_with_subplots) 
fig_dims = (3, 1)

# Plot the AgeFill histogram for Survivors
plt.subplot2grid(fig_dims, (0, 0))
survived_df = df_train[df_train['Survived'] == 1]
survived_df['AgeFill'].hist(bins=max_age / bin_size, range=(1, max_age))

# Plot the AgeFill histogram for Females
plt.subplot2grid(fig_dims, (1, 0))
females_df = df_train[(df_train['Sex_val'] == 0) & (df_train['Survived'] == 1)]
females_df['AgeFill'].hist(bins=max_age / bin_size, range=(1, max_age))

# Plot the AgeFill histogram for first class passengers
plt.subplot2grid(fig_dims, (2, 0))
class1_df = df_train[(df_train['Pclass'] == 1) & (df_train['Survived'] == 1)]
class1_df['AgeFill'].hist(bins=max_age / bin_size, range=(1, max_age))

#5.Feature: Family Size
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch']
df_train['FamilySize'].hist()
plt.title('Family Size Histogram')

family_sizes = sorted(df_train['FamilySize'].unique())
family_size_max = max(family_sizes)

df3 = df_train[df_train['Survived'] == 0]['FamilySize']
df4 = df_train[df_train['Survived'] == 1]['FamilySize']
plt.hist([df3, df4], 
         bins=family_size_max + 1, 
         range=(0, family_size_max), 
         stacked=True)
plt.legend(('Died', 'Survived'), loc='best')
plt.title('Survivors by Family Size')

#6.Final Data Preparation for Machine Learning
df_train.dtypes[df_train.dtypes.map(lambda x: x == 'object')]
df_train2 = df_train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
df_train2 = df_train2.drop(['Age', 'SibSp', 'Parch', 'PassengerId', 'Embarked_val'], axis=1)
train_data = df_train2.values #Convert the DataFrame to a numpy array








