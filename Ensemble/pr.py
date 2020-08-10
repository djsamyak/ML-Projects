import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

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
X = dataset.iloc[131,4:].index      #days
y = dataset.iloc[131,4:].values     #Confimed Patients

numericValue_tenDaysAhead = len(y)+10
tenDays = np.array([range(len(y)+10)])
tenDaysAhead = np.reshape(tenDays,(-1, 1))

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

XMat = np.reshape(X,(-1, 1))
yMat = np.reshape(y,(-1, 1))

#Polynomial Regression
polyRegressor = PolynomialFeatures(degree = 11)
Xpoly = polyRegressor.fit_transform(XMat)
linRegressor2 = LinearRegression()
linRegressor2.fit(Xpoly,yMat)
prediction = abs(linRegressor2.predict(polyRegressor.fit_transform(tenDaysAhead)))

print(f"RMSE = {sqrt(mean_squared_error(yMat,linRegressor2.predict(polyRegressor.fit_transform(XMat))))}")
print(f"MSE = {mean_squared_error(yMat,linRegressor2.predict(polyRegressor.fit_transform(XMat)))}")
print(f"MAE = {mean_absolute_error(yMat,linRegressor2.predict(polyRegressor.fit_transform(XMat)))}")
print(f"MAPE = {mean_absolute_percentage_error(yMat,linRegressor2.predict(polyRegressor.fit_transform(XMat)))}")

ydata = prediction


