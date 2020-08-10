import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from dateplot import *
import datetime

def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100

dataset = pd.read_csv(r"C:\Users\djsam\Desktop\ML\COVID19_Data\COVID-19\csse_covid_19_data\csse_covid_19_time_series\time_series_covid19_confirmed_global_new1.csv")
X = list(dataset.columns)
X = X[4:]                               #days
y = dataset.iloc[131:132,4:].values     #Confimed Patients

X = np.array(X)
X = X.astype(int)
X = np.reshape(X,(-1, 1))
y = np.reshape(y,(-1, 1))               #Converted y to appropriate form  - y.transpose()

gsc = GridSearchCV(SVR(),{
    "kernel":['poly'],
    "degree":[5,6,7,8],
    "C":[1.9,2,2.1,2.2],
    "epsilon":[0.07,0.08,0.09,0.1]},
    #"coef0":[0,0.2,0.3,0.5,0.7,0.9]},
    cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
grid_result = gsc.fit(X,y.ravel())
params = grid_result.best_params_

model = SVR(kernel = "poly",degree = 6, epsilon=99, C=1)
model.fit(X,y.ravel())
predited_values = model.predict(X)

print(f"\nRMSE = {mean_squared_error(y,predited_values,squared = False)}")
print(f"MSE = {mean_squared_error(y,predited_values,squared = True)}")
print(f"MAE = {mean_absolute_error(y,predited_values)}")
print(f"MAPE = {mean_absolute_percentage_error(y,predited_values)}")

X_ahead = []
for i in range(len(X)+10):
    X_ahead.append(i)
X_ahead = np.reshape(X_ahead,(-1, 1))

model = SVR(kernel = "poly",degree = 6, epsilon=99, C=1)
model.fit(X,y.ravel())
predited_values1 = model.predict(X_ahead)

dateplot(X,predited_values,predited_values1)
