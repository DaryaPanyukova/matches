#!/usr/bin/env python
# coding: utf-8

# In[689]:


import pandas as pd
import numpy as np
import datetime
import time
from sklearn.metrics import mean_absolute_error
import random
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit


# In[730]:


df = pd.read_csv(r"C:\Hahaton\season-1920_csv.csv")


# In[731]:


del_columns = ['Time', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA',
               'VCH', 'VCD', 'VCA', 'MaxH', 'MaxD', 'MaxA', 'AvgH','AvgD', 'AvgA', 'B365>2.5', 'B365<2.5', 'P>2.5', 'P<2.5', 
               'Max>2.5', 'Max<2.5', 'Avg>2.5', 'Avg<2.5', 'AHh','B365AHH', 'B365AHA', 'PAHH', 'PAHA', 'MaxAHH', 'MaxAHA',
               'AvgAHH', 'AvgAHA', 'B365CH', 'B365CD', 'B365CA', 'BWCH','BWCD', 'BWCA', 'IWCH', 'IWCD', 'IWCA', 'PSCH', 
               'PSCD', 'PSCA', 'WHCH', 'WHCD', 'WHCA', 'VCCH', 'VCCD', 'VCCA', 'MaxCH', 'MaxCD', 'MaxCA', 'AvgCH', 'AvgCD', 
               'AvgCA', 'B365C>2.5',   'B365C<2.5', 'PC>2.5', 'PC<2.5', 'MaxC>2.5', 'MaxC<2.5','AvgC>2.5', 'AvgC<2.5', 'AHCh', 
               'B365CAHH', 'B365CAHA','PCAHH', 'PCAHA', 'MaxCAHH','MaxCAHA', 'AvgCAHH', 'AvgCAHA'] 
df = df.drop(del_columns, axis = 1)


# In[732]:


df.Date = df.Date.apply(lambda x: time.mktime(datetime.datetime.strptime(x, "%d/%m/%Y").timetuple()))


# In[733]:


df = df.sort_values(by = ['Date'])
df.head(5)


# In[734]:


def result(x):
    if x == 'D':
        return 0
    elif x == 'A':
        return 2
    
    return 1
df.FTR = df.FTR.apply(result)
df.HTR = df.HTR.apply(result)


# In[737]:


df["HomeHitAccuracy"] = df["HS"] / df["HST"]
df["AwayHitAccuracy"] = df["AS"] / df["AST"]
df["HomeFouls"] = df["HF"] + df["HY"] + df["HR"]
df["AwayFouls"] = df["AF"] + df["AY"] + df["AR"]

df['ratio_goals'] = df['FTHG']-df['FTAG']


# In[738]:


df = get_rolling_features(df, depth).dropna()


# In[739]:


depth = 5

useful_features = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC',
                  'HY', 'AY', 'HR', 'AR', 'HomeHitAccuracy', 'AwayHitAccuracy', 'HomeFouls', 'AwayFouls', 'ratio_goals', 'FTR']

cat_features = []

for j in ['HomeTeam', 'AwayTeam']:
    for k in range(1, depth+1):
        cat_features.append(f'prev_FTR_{j}_{k}')
    
print(cat_features)



def get_rolling_features(df, depth):  #Проверить
    for i in useful_features:
        for j in ['HomeTeam', 'AwayTeam']:
            for k in range(1, depth+1):
                df[f'prev_{i}_{j}_{k}'] = df.groupby(j)[i].shift(k)
                
                

    return df

def get_rolling_statistics(df, depth): # Проверить
    for i in useful_features[-1:]: #drop FTR
        for j in ['HomeTeam', 'AwayTeam']:
            feature_names = [f'prev_{i}_{j}_{k}' for k in range(1,depth+1)]

            df[f'mean_{i}_{j}_3'] = df[feature_names[-3:]].mean(axis=1)
            df[f'mean_{i}_{j}_5'] = df[feature_names].mean(axis=1)
            
            df[f'quantile25_{i}_{j}_3'] = df[feature_names].apply(lambda x: np.quantile(x[-3:], 0.25),axis=1)
            df[f'quantile25_{i}_{j}_5'] = df[feature_names].apply(lambda x: np.quantile(x, 0.25), axis=1)
            
            
            df[f'quantile75_{i}_{j}_3'] = df[feature_names].apply(lambda x: np.quantile(x[-3:], 0.75), axis=1)
            df[f'quantile75_{i}_{j}_5'] = df[feature_names].apply(lambda x: np.quantile(x, 0.75), axis=1)
            
            df[f'std_{i}_{j}_5'] = df[feature_names].std(axis=1)
            
            df[f'max_{i}_{j}_3'] = df[feature_names[-3:]].max(axis=1)
            df[f'max_{i}_{j}_5'] = df[feature_names].max(axis=1)
            
            df[f'min_{i}_{j}_3'] = df[feature_names[-3:]].min(axis=1)
            df[f'min_{i}_{j}_5'] = df[feature_names].min(axis=1)
            
            df[f'mid_range_{i}_{j}_3'] = 0.5*(df[f'min_{i}_{j}_3']+df[f'max_{i}_{j}_3'])
            df[f'mid_range_{i}_{j}_5'] = 0.5*(df[f'min_{i}_{j}_5']+df[f'max_{i}_{j}_5'])
            
            df[f'std_{i}_{j}_3'] = df[feature_names[-3:]].std(axis=1)
            df[f'std_{i}_{j}_5'] = df[feature_names].min(axis=1)
            
            df[f'var_coef_{i}_{j}_3'] = (df[f'std_{i}_{j}_3']**2)/df[f'mean_{i}_{j}_3']
            df[f'var_coef_{i}_{j}_5'] = (df[f'std_{i}_{j}_5']**2)/df[f'mean_{i}_{j}_5']
            
            #df['35'] = df[feature_names].apply(lambda x: len(x),axis=1)
            
            #df[f'mean_{i}_{j}_5'] = df['q25_5+']+df['q75_5']
            
            #df[f'mid_range_{i}_{j}_3'] = df[feature_names[:3]].apply(lambda x: 0.5 * (min(df[feature_names[:3]])+max(df[feature_names[:3]])))
            #df[f'mid_range_{i}_{j}_5'] = df[feature_names].mean(axis=1)


    return df
        


# In[740]:


df = get_rolling_statistics(df, depth)


# In[741]:


X.shape


# In[742]:



df = df.drop(["HomeTeam", "AwayTeam"], axis = 1)


# In[743]:


unuseful_features = ['HomeFouls','HTHG', 'HTAG', 'HomeHitAccuracy', 'AwayHitAccuracy', 'HTR', 'Div', 'Date','FTHG', 'FTAG', 
                     'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 
                     'AR', 'FTR', 'Referee', 'AwayFouls', 'ratio_goals']
X = df.drop(unuseful_features, axis = 1)
y = df.FTR


# In[704]:


# for i,j in zip(cboost.feature_importances_, cboost.feature_names_):
#     if i/sum(cboost.feature_importances_)>0.01:
#         print(j,i/sum(cboost.feature_importances_))


# In[705]:


# Разница голов/кол-во голов как таргет
# регрессия которая предсказывает разницу голов/квадрат разницы голов
# Телеграм формула дисперсии для разницы голов
# параметры catboost для кроссвалидации
# Линейную модель вместо катбуста, рэндом форест


# In[750]:


cboost = CatBoostClassifier(
    iterations= 300,
    depth = 5,
    learning_rate = 0.01,
    cat_features = cat_features,
    verbose=500,
)


# In[752]:


X[cat_features] = X[cat_features].astype(str)


# In[753]:


from sklearn.metrics import accuracy_score
n_splits=6
tscv = TimeSeriesSplit(n_splits=n_splits)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    cboost.fit(X_train, y_train)
    train_score = cboost.score(X_train, y_train)
    test_score = cboost.score(X_test, y_test)
    
    print(test_score, train_score)
    


# In[754]:


for i,j in zip(cboost.feature_importances_, cboost.feature_names_):
    if i/sum(cboost.feature_importances_)>0.007:
        print(j,i/sum(cboost.feature_importances_))


# In[761]:


sum_all_features = sum(cboost.feature_importances_)
norm_features = [i/sum_all_features for i in cboost.feature_importances_]
sorted_features = zip(norm_features, cboost.feature_names_)
sorted_features = sorted(sorted_features, key=lambda x: x[0])


# In[762]:


for i in sorted_features[::-1]:
    print(i)

