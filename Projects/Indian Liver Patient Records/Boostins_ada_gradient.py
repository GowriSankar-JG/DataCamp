# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 00:34:02 2020

@author: Gowrisankar JG
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ds=pd.read_csv("indian_liver_patient_preprocessed.csv")

X=ds.iloc[:,:10]
y=ds.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111)

dt = DecisionTreeClassifier(max_depth=2, random_state=1)
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=180, random_state=1)

ada.fit(X_train, y_train)
y_pred_proba = ada.predict_proba(X_test)[:,1]

ada_roc_auc = roc_auc_score(y_test, y_pred_proba)
print('ROC AUC score: {:.2f}'.format(ada_roc_auc))

####################################################################################################

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as MSE

gb = GradientBoostingRegressor(max_depth=4,n_estimators=200,random_state=2)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)

mse_test = MSE(y_test, y_pred)
rmse_test = mse_test**(1/2)
print('Test set RMSE of gb: {:.3f}'.format(rmse_test))

####################################################################################################

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as MSE

sgbr = GradientBoostingRegressor(max_depth=4, subsample=0.9, max_features=0.75, 
                                 n_estimators=200, random_state=2)

sgbr.fit(X_train, y_train)
y_pred = sgbr.predict(X_test)
mse_test = MSE(y_test, y_pred)
rmse_test = mse_test**(1/2)

print('Test set RMSE of sgbr: {:.3f}'.format(rmse_test))



