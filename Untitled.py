#!/usr/bin/env python
# coding: utf-8

# In[533]:


import pandas as pd
import numpy as np
import datetime
import time
from sklearn.metrics import mean_absolute_error
import random
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit


# In[534]:


df = pd.read_csv(r"C:\Hahaton\season-1920_csv.csv")


# In[535]:


del_columns = ['Time', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA',
               'VCH', 'VCD', 'VCA', 'MaxH', 'MaxD', 'MaxA', 'AvgH','AvgD', 'AvgA', 'B365>2.5', 'B365<2.5', 'P>2.5', 'P<2.5', 
               'Max>2.5', 'Max<2.5', 'Avg>2.5', 'Avg<2.5', 'AHh','B365AHH', 'B365AHA', 'PAHH', 'PAHA', 'MaxAHH', 'MaxAHA',
               'AvgAHH', 'AvgAHA', 'B365CH', 'B365CD', 'B365CA', 'BWCH','BWCD', 'BWCA', 'IWCH', 'IWCD', 'IWCA', 'PSCH', 
               'PSCD', 'PSCA', 'WHCH', 'WHCD', 'WHCA', 'VCCH', 'VCCD', 'VCCA', 'MaxCH', 'MaxCD', 'MaxCA', 'AvgCH', 'AvgCD', 
               'AvgCA', 'B365C>2.5',   'B365C<2.5', 'PC>2.5', 'PC<2.5', 'MaxC>2.5', 'MaxC<2.5','AvgC>2.5', 'AvgC<2.5', 'AHCh', 
               'B365CAHH', 'B365CAHA','PCAHH', 'PCAHA', 'MaxCAHH','MaxCAHA', 'AvgCAHH', 'AvgCAHA'] 
df = df.drop(del_columns, axis = 1)


# In[536]:


df.Date = df.Date.apply(lambda x: time.mktime(datetime.datetime.strptime(x, "%d/%m/%Y").timetuple()))


# In[537]:


df = df.sort_values(by = ['Date'])
df.Date


# In[519]:


def result(x):
    if x == 'D':
        return 0
    elif x == 'A':
        return 2
    
    return 1
df.FTR = df.FTR.apply(result)
df.HTR = df.HTR.apply(result)


# In[520]:


df["HomeHitAccuracy"] = df["HS"] / df["HST"]
df["AwayHitAccuracy"] = df["AS"] / df["AST"]
df["HomeFouls"] = df["HF"] + df["HY"] + df["HR"]
df["AwayFouls"] = df["AF"] + df["AY"] + df["AR"]


# In[521]:


useful_features = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC',
                  'HY', 'AY', 'HR', 'AR', 'HomeHitAccuracy', 'AwayHitAccuracy', 'HomeFouls', 'AwayFouls']

depth = 5

def get_rolling_features(df, depth):
    for i in useful_features:
        for j in ['HomeTeam', 'AwayTeam']:
            for k in range(1, depth+1):
                df[f'prev_{i}_{j}_{k}'] = df.groupby(j)[i].shift(k)

    return df

def get_rolling_statistics(df, depth):
    for i in useful_features:
        for j in ['HomeTeam', 'AwayTeam']:
            feature_names = [f'prev_{i}_{j}_{k}' for k in range(1,depth+1)]
            print(feature_names)
            df[f'mean_{i}_{j}_3'] = df[feature_names[:3]].mean(axis=1)
            df[f'mean_{i}_{j}_5'] = df[feature_names].mean(axis=1)
            
            df[f'quantile25_{i}_{j}_3'] = df[feature_names].apply(lambda x: np.quantile(x[:3], 0.25),axis=1)
            df[f'quantile25_{i}_{j}_5'] = df[feature_names].apply(lambda x: np.quantile(x, 0.25), axis=1)
            
            
            df[f'quantile75_{i}_{j}_3'] = df[feature_names].apply(lambda x: np.quantile(x[:3], 0.75), axis=1)
            df[f'quantile75_{i}_{j}_5'] = df[feature_names].apply(lambda x: np.quantile(x, 0.75), axis=1)
            
            #df['35'] = df[feature_names].apply(lambda x: len(x),axis=1)
            
            #df[f'mean_{i}_{j}_5'] = df['q25_5+']+df['q75_5']
            
            #df[f'mid_range_{i}_{j}_3'] = df[feature_names[:3]].apply(lambda x: 0.5 * (min(df[feature_names[:3]])+max(df[feature_names[:3]])))
            #df[f'mid_range_{i}_{j}_5'] = df[feature_names].mean(axis=1)


    return df
        


# In[522]:


df = get_rolling_features(df, depth).dropna()


# In[523]:


df1 = get_rolling_statistics(df, depth)
# mean_AY_AwayTeam_3
df1.columns


# In[524]:


X.head(5)


# In[525]:



df = df.drop(["HomeTeam", "AwayTeam"], axis = 1)


# In[526]:


unuseful_features = ['HomeFouls','HTHG', 'HTAG', 'HomeHitAccuracy', 'AwayHitAccuracy', 'HTR', 'Div', 'Date','FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 
                     'AR', 'FTR', 'Referee', 'AwayFouls']
X = df.drop(unuseful_features, axis = 1)
y = df.FTR
X.columns


# In[527]:


cboost = CatBoostClassifier(
    iterations= 200,
    depth = 5,
    learning_rate = 0.01,
    #cat_features = ["HomeTeam", "AwayTeam"],
    verbose=500,
)


# In[528]:


n_splits=5
tscv = TimeSeriesSplit(n_splits=n_splits)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    cboost.fit(X_train, y_train)
    train_score = cboost.score(X_train, y_train)
    test_score = cboost.score(X_test, y_test)
    
    print(test_score, train_score)
    


# In[532]:


train_score = cboost.score(X_train, y_train)
test_score = cboost.score(X_test, y_test)
print(test_score, train_score)


# In[337]:


#list(zip(X.columns, cboost.get_feature_importance()))


# In[531]:


for i,j in zip(cboost.feature_importances_, cboost.feature_names_):
    if i/sum(cboost.feature_importances_)>0.01:
        print(j,i/sum(cboost.feature_importances_))


# In[ ]:




