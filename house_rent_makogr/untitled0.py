# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 19:28:48 2022

@author: safak
"""

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,ShuffleSplit,GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
from sklearn.svm import SVR

path="datasets"

df=pd.read_csv(path+"\\House_Rent_Dataset.csv")

df=df.drop(["Posted On"],axis=1)
df=df.drop(["Point of Contact"],axis=1)
df=df.drop(["Area Locality"],axis=1)
df=df.drop(["Floor"],axis=1)
df=df.dropna()

df_new=pd.get_dummies(df,columns=["BHK"],prefix=["BHK"])
df_new=pd.get_dummies(df_new,columns=["Area Type"],prefix=["Area Type"])
df_new=pd.get_dummies(df_new,columns=["City"],prefix=["City"])
df_new=pd.get_dummies(df_new,columns=["Furnishing Status"],prefix=["Furnishing Status"])
df_new=pd.get_dummies(df_new,columns=["Tenant Preferred"],prefix=["Tenant Preferred"])

y_datas=df_new["Rent"]
df_new=df_new.drop(["Rent"],axis=1)

# splitting the datas
x_train,x_test,y_train,y_test=train_test_split(df_new,y_datas,test_size=0.25,random_state=42)

scaler=StandardScaler()
scaler.fit(x_train)

x_train_scaled=scaler.transform(x_train)
x_test_scaled=scaler.transform(x_test)

# machine learning part (artifical neural network)
"""mlp_model=MLPRegressor().fit(x_train_scaled,y_train)

mlp_params={"alpha":[0.1,0.01,0.02,0.03,0.2,0.002,0.003,0.005],
            "activation":["relu","logistic"]}

mlp_cv_model=GridSearchCV(mlp_model,mlp_params,cv=10)
mlp_cv_model.fit(x_train_scaled,y_train)

mlp_tuned=MLPRegressor(alpha=0.005,activation="relu")
mlp_tuned.fit(x_train_scaled,y_train)"""

# (coklu lineer regresyon)
"""lm=LinearRegression()
model=lm.fit(x_train,y_train)

my_model=pickle.load(open("hind_house.pickle","rb"))"""

# (radyal SVR)
svr_model=SVR("rbf").fit(x_train,y_train)

svr_params={"C":[0.1,0.2,0.4,0.8,3,4,5,10,15,20,25,30,40,50]}
svr_cv_model=GridSearchCV(svr_model, svr_params,cv=10)
svr_cv_model.fit(x_train,y_train)

svr_tuned=SVR("rbf",C=pd.Series(svr_cv_model.best_params_)[0]).fit(x_train,y_train)

model_name="hind_house_svr.pickle"
pickle.dump(svr_tuned,open(model_name,"wb"))

my_model=pickle.load(open("hind_house_svr.pickle","rb"))

print("hata skoru: {}".format(np.sqrt(-cross_val_score(my_model,x_test_scaled,y_test,cv=10,scoring="neg_mean_squared_error")).mean()))

x_test.index=np.arange(0,len(x_test),1)
y_test.index=np.arange(0,len(y_test),1)

ind=7
sonuc=x_test.iloc[ind]
sonuc=pd.DataFrame(sonuc).T

print("deneme -> gercek deger: {} |-|-|-| tahmin degeri: {}".format(y_test[ind],my_model.predict(sonuc)))
    



















































