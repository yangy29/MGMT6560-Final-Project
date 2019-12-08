# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:11:47 2019

@author: yolan
"""

import pandas as pd 
import numpy as np
import gc
from sklearn.model_selection import train_test_split

#Loading data--------------------------
building_metadata = pd.read_csv('building_metadata.csv')
train = pd.read_csv('train.csv')
weather_train = pd.read_csv('weather_train.csv')
#test = pd.read_csv('test.csv')
#weather_test = pd.read_csv('weather_test.csv')

#Merge everything into train & test dataset
train = train.merge(building_metadata, on='building_id', how='left')
#test = test.merge(building_metadata, on='building_id', how='left')
del building_metadata
train = train.merge(weather_train, on=['site_id', 'timestamp'], how='left')
#test = test.merge(weather_test, on=['site_id', 'timestamp'], how='left')
del weather_train
gc.collect()

#Due to the ram limitation, I did a reduction of the rows to make it easier to run
train = train.sample(frac=0.01)
#test = test.sample(frac=0.01)
#----------------------------------------------------------
#PCA to select the several principal component
#from sklearn.decomposition import PCA
#pca = PCA(n_components =2 )
#X_pca = pca.fit_transform(train)
#--------------------------------------------------------------

# Saving some memory
d_types = {'building_id': np.int16,
          'meter': np.int8,
          'site_id': np.int8,
          'primary_use': 'category',
          'square_feet': np.int32,
          'year_built': np.float16,
          'floor_count': np.float16,
          'air_temperature': np.float32,
          'cloud_coverage': np.float16,
          'dew_temperature': np.float32,
          'precip_depth_1_hr': np.float16,
          'sea_level_pressure': np.float32,
          'wind_direction': np.float16,
          'wind_speed': np.float32}

for feature in d_types:
    train[feature] = train[feature].astype(d_types[feature])
    #test[feature] = test[feature].astype(d_types[feature])
    
train["timestamp"] = pd.to_datetime(train["timestamp"], format="%Y-%m-%d %H:%M:%S")
#test["timestamp"] = pd.to_datetime(test["timestamp"], format="%Y-%m-%d %H:%M:%S")
gc.collect()

#So far raw data has been prepared-----------------------------------------------------




##Preparing data--------------------------------------------------------
def data_prep(data):
    
    #del data['floor_count'], data['precip_depth_1_hr'], data['primary_use'], data['year_built'], data['cloud_coverage']
    
    #data['date'] = data.timestamp.dt.date
    data["hour"] = data.timestamp.dt.hour
    data['month'] = data.timestamp.dt.month   #####
    data["weekday"] = data.timestamp.dt.weekday
    data.loc[data['weekday'] <= 4,"weekday"] = 1
    data.loc[data['weekday'] > 4,"weekday"] = 0
    
    data.square_feet = np.log1p(data.square_feet)
    
    del data["timestamp"]
    #data.fillna(0)
    
    return data
#---------------------------------------------------------
train = data_prep(train)

X_train = train.drop(['meter_reading'], axis=1)
y_train = np.log1p(train['meter_reading'])

train_X, test_X, train_y,test_y = train_test_split(X_train, y_train,test_size=0.1, random_state=100)

#test = data_prep(test)
#X_test = test.drop(['row_id'], axis = 1)
#test_id = test[['row_id']]

back_train = train
#back_test = test
del train

gc.collect()



## Model setting up ---------------------------------

import lightgbm as lgb
from sklearn.metrics import mean_squared_log_error



categorical_features = ["building_id", "site_id", "meter", "month","hour", "weekday","wind_direction", "year_built"]

train_data = lgb.Dataset(train_X, label=train_y, 
                         categorical_feature=["building_id", "site_id", "meter", "month",
                                              "hour", "weekday","wind_direction", "year_built"])


params = {
    "objective": "regression",
    "boosting": "gbdt",
    "num_leaves": 40,
    "learning_rate": 0.05,
    "feature_fraction": 0.85,
    "reg_lambda": 2,
    "metric": "rmsle"
}

print("Building model with lgb:")
model= lgb.train(params, train_set=train_data, num_boost_round=100, verbose_eval=200)

pred_y = model.predict(test_X)

#---------Check

#rmsle1 = np.sqrt(mean_squared_log_error(y_train,y_pred))
rmsle2 = np.sqrt(mean_squared_log_error(test_y,pred_y))


#-----------plot----------
import seaborn as sns
from matplotlib import pyplot as plt


df_fimp = pd.DataFrame()
df_fimp["feature"] = train_X.columns.values
df_fimp["importance"] = model.feature_importance()
df_fimp["half"] = 1

plt.figure(figsize=(14, 7))
sns.barplot(x="importance", y="feature", data=df_fimp.sort_values(by="importance", ascending=False))
plt.title("LightGBM Feature Importance")
plt.tight_layout()























