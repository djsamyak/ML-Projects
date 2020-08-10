import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import datetime

import warnings
warnings.filterwarnings("ignore")

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

dataset = pd.read_csv(r"C:\Users\djsam\Desktop\ML\Dataset\covid19.csv")
X = list(dataset.columns)
X = X[4:]                               #days
y = dataset.iloc[131:132,4:].values     #Confimed Patients

X = np.array(X)
X = X.astype(int)
X = np.reshape(X,(-1, 1))
y = np.reshape(y,(-1, 1))               #Converted y to appropriate form  - y.transpose()

# for i in range(len(y)):
#     if y[i,0] == 0:
#         y[i,0] = 1

model = ExponentialSmoothing(y,trend="add",damped = True, seasonal = None)
model_fit = model.fit(0.3,0.1)
yhat = model_fit.predict(start = 1, end = len(y))
yhat = np.reshape(yhat,(-1, 1))

# gsc = GridSearchCV(ExponentialSmoothing(y,trend="add",damped = True, seasonal = None),
#                                         {"trend" : ["add", "mul", "additive", "multiplicative"],
#                                          "damped": [True,False]},
#                                         cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

# grid_result = gsc.fit(X,y.ravel())
# best_params = grid_result.best_params_

model = ExponentialSmoothing(y,trend="add",damped = True, seasonal = None)
model_fit = model.fit(0.3,0.1)
yhat1 = model_fit.predict(start = 1, end = len(y)+10)
yhat1 = np.reshape(yhat1,(-1, 1))

print(f"RMSE = {mean_squared_error(y,yhat,squared = False)}")
print(f"MSE = {mean_squared_error(y,yhat,squared = True)}")
print(f"MAE = {mean_absolute_error(y,yhat)}")
print(f"MAPE = {mean_absolute_percentage_error(y,yhat)} ")

dates = []
for i in range(len(yhat)):
    dates.append(datetime.date(2020,1,22) + datetime.timedelta(i))

dates10ahead = []
for i in range(len(yhat)+10):
    dates10ahead.append(datetime.date(2020,1,22) + datetime.timedelta(i))
