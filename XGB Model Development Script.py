# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import pickle

from sklearn import metrics
import sklearn.model_selection as model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor

# Read the data from the Excel file

##df1 = pd.read_excel(r"C:\Users\Administrator\Desktop\HSM Property Prediction Model\HSM2\datasets\HSM2 (01) - Jan20 to db.xlsx")
##df2 = pd.read_csv(r"C:\Users\Administrator\Desktop\HSM Property Prediction Model\HSM2\datasets\hsm2(db to 15July21).csv")
##df2 = df2.drop(['COIL_GEN_TIME'], axis = 1)
##
##df=pd.concat([df1, df2], axis=0, sort=False)
##
##df.to_excel(r"C:\Users\Administrator\Desktop\HSM Property Prediction Model\HSM2\datasets\hsm2 dataset.xlsx", index=False, header=True)
##df=df2
df = pd.read_excel(r"C:\Users\research.modeling\Desktop\Prediction Models\HSM Property Prediction Model\models\HSM-2.xlsx","model")
#df = pd.read_excel(r"C:\Users\prana\Desktop\HSM 1\HSM1 full.xlsx","WT02A00")
#df = pd.read_excel(r"D:\oracle\deployment\HSM1\SP3.xlsx")
#df = pd.read_excel(r"D:\oracle\deployment\HSM1\data 2021.xlsx")

print("##################### Generating XGboost Model #####################")

# Initialize Features

#X = df.drop(['COIL_ID', 'GRADE',  'UTS','YS','P_ELONGATION'], axis = 1)# Initialize Target
X = df.drop(['COIL_ID', 'GRADE', 'COIL_GEN_TIME', 'UTS', 'YS', 'P_ELONGATION', 'UTS_PRED', 'YS_PRED', 'EL_PRED'], axis = 1)# Initialize Target

Y1 = df['UTS']
Y2 = df['YS']
Y3 = df['P_ELONGATION']

ts=0.3
rs1=2
rs2=0

X1_train, X1_test, Y1_train, Y1_test = model_selection.train_test_split(X,Y1,test_size=ts,random_state=rs1)
X2_train, X2_test, Y2_train, Y2_test = model_selection.train_test_split(X,Y2,test_size=ts,random_state=rs1)
X3_train, X3_test, Y3_train, Y3_test = model_selection.train_test_split(X,Y3,test_size=ts,random_state=rs1)

# define model
##model_uts = XGBRegressor(n_estimators=150, max_depth=8, eta=0.1, nthread=-1)
##model_ys = XGBRegressor(n_estimators=258, max_depth=8, eta=0.1, nthread=-1)
##model_el = XGBRegressor(n_estimators=100, max_depth=8, eta=0.1, nthread=-1)

# define model
model_uts = XGBRegressor(n_estimators=261, max_depth=12, eta=0.05, subsample=0.7, colsample_bytree=1, nthread=-1)
model_ys = XGBRegressor(n_estimators=246, max_depth=14, eta=0.05, subsample=0.7, colsample_bytree=1, nthread=-1)
model_el = XGBRegressor(n_estimators=246, max_depth=9, eta=0.04, subsample=0.7, colsample_bytree=1, nthread=-1)

##model_uts = XGBRegressor(booster='gbtree', gamma=0, n_estimators=250, max_depth=14, eta=0.05, subsample=0.8, colsample_bytree=1, nthread=-1)
##model_ys = XGBRegressor(booster='gbtree', gamma=0, n_estimators=168, max_depth=13, eta=0.09, subsample=0.7, colsample_bytree=0.8, nthread=-1)
##model_el = XGBRegressor(booster='gbtree', gamma=0, n_estimators=161, max_depth=12, eta=0.04, subsample=0.9, colsample_bytree=1, nthread=-1)

##model_uts = XGBRegressor(booster='gbtree', gamma=0, n_estimators=250, max_depth=14, eta=0.05, subsample=0.8, colsample_bytree=1, nthread=-1)
##model_ys = XGBRegressor(booster='gbtree', gamma=0, n_estimators=8600, max_depth=13, eta=0.03, subsample=0.7, colsample_bytree=0.8, nthread=-1)
##model_el = XGBRegressor(booster='gbtree', gamma=0, n_estimators=161, max_depth=12, eta=0.04, subsample=0.9, colsample_bytree=1, nthread=-1)

#model_uts = XGBRegressor(n_estimators=660, max_depth=12, eta=0.1, subsample=0.7, colsample_bytree=1)
#model_uts = XGBRegressor(n_estimators=150,max_depth=12, eta=0.1, subsample=0.7, colsample_bytree=1, nthread=-1)
#model_ys = XGBRegressor(n_estimators=260,max_depth=12, eta=0.1, subsample=0.8, colsample_bytree=1, nthread=-1)
#model_el = XGBRegressor(n_estimators=100,max_depth=12, eta=0.1, colsample_bytree=1, nthread=-1)
##### Finalized
##model_uts = XGBRegressor(n_estimators=150, max_depth=12, eta=0.1, subsample=0.7, colsample_bytree=1)
##model_ys = XGBRegressor(n_estimators=260, max_depth=12, eta=0.1, subsample=0.8, colsample_bytree=1)
##model_pe = XGBRegressor(n_estimators=76, max_depth=12, eta=0.1, subsample=0.7, colsample_bytree=1)

model_uts.fit(X1_train, Y1_train)#, eval_set=[(X1_train, Y1_train), (X1_test , Y1_test)], eval_metric=['rmsle'])
model_ys.fit(X2_train, Y2_train)
model_el.fit(X3_train, Y3_train)

pred_uts = model_uts.predict(X1_test)
pred_ys = model_ys.predict(X2_test)
pred_el = model_el.predict(X3_test)

pickle.dump(model_uts, open('UTS_XGB.pkl', 'wb'))
pickle.dump(model_ys, open('YS_XGB.pkl', 'wb'))
pickle.dump(model_el, open('EL_XGB.pkl', 'wb'))

#abs_uts = mean_absolute_error(Y1_test, y_pred_uts)
#sq_uts = mean_squared_error(Y1_test, y_pred_uts)

print('UTS R2:', np.sqrt(metrics.r2_score(Y1_test, pred_uts)))
print('YS R2:', np.sqrt(metrics.r2_score(Y2_test, pred_ys)))
print('EL R2:', np.sqrt(metrics.r2_score(Y3_test, pred_el)))

print("XGB model generated !!!")
