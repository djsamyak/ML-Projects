import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

tenDays = np.array([range(131)])
tenDaysAhead = np.reshape(tenDays,(-1, 1))

date = [1/22/20,1/23/20,1/24/20,1/25/20,1/26/20,1/27/20,1/28/20,1/29/20,1/30/20,1/31/20,2/1/20,2/2/20,2/3/20,2/4/20,2/5/20,2/6/20,2/7/20,2/8/20,2/9/20,2/10/20,2/11/20,2/12/20,2/13/20,2/14/20,2/15/20,2/16/20,2/17/20,2/18/20,2/19/20,2/20/20,2/21/20,2/22/20,2/23/20,2/24/20,2/25/20,2/26/20,2/27/20,2/28/20,2/29/20,3/1/20,3/2/20,3/3/20,3/4/20,3/5/20,3/6/20,3/7/20,3/8/20,3/9/20,3/10/20,3/11/20,3/12/20,3/13/20,3/14/20,3/15/20,3/16/20,3/17/20,3/18/20,3/19/20,3/20/20,3/21/20,3/22/20,3/23/20,3/24/20,3/25/20,3/26/20,3/27/20,3/28/20,3/29/20,3/30/20,3/31/20,4/1/20,4/2/20,4/3/20,4/4/20,4/5/20,4/6/20,4/7/20,4/8/20,4/9/20,4/10/20,4/11/20,4/12/20,4/13/20,4/14/20,4/15/20,4/16/20,4/17/20,4/18/20,4/19/20,4/20/20,4/21/20,4/22/20,4/23/20,4/24/20,4/25/20,4/26/20,4/27/20,4/28/20,4/29/20,4/30/20,5/1/20,5/2/20,5/3/20,5/4/20,5/5/20,5/6/20,5/7/20,5/8/20,5/9/20,5/10/20,5/11/20,5/12/20,5/13/20,5/14/20,5/15/20,5/16/20,5/17/20,5/18/20,5/19/20,5/20/20,5/21/20]
dates = np.reshape(date,(-1, 1))

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

dataset = pd.read_csv(r"C:\Users\djsam\Desktop\COVID19_Data\COVID-19\csse_covid_19_data\csse_covid_19_time_series\time_series_covid19_confirmed_global.csv")
X = dataset.iloc[131,4:].index      #days
y = dataset.iloc[131,4:].values     #Confimed Patients


# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

XMat = np.reshape(X,(-1, 1))
yMat = np.reshape(y,(-1, 1))

#Linear Regression
linRegressor = LinearRegression()
linRegressor.fit(XMat,yMat)
prediction = linRegressor.predict(XMat)+2.13926247e+04

#Polynomial Regression
polyRegressor = PolynomialFeatures(degree = 17)
Xpoly = polyRegressor.fit_transform(XMat)
linRegressor2 = LinearRegression()
linRegressor2.fit(Xpoly,yMat)

#Linear Reg Plot
plt.xticks(np.arange(0, 121, step=10))
plt.scatter(X,y,color="red",marker=".")
plt.plot(X,prediction,color="blue")
plt.title("Growth of COVID-19 through 22nd January, 2020 in India (Linear)")
plt.xlabel("Days since 22nd January, 2020")
plt.ylabel("Total confirmed cases")

print(f"MSE with Linear Regression: {mean_squared_error(yMat,linRegressor.predict(XMat))}")
print(f"RMSE with Linear Regression: {sqrt(mean_squared_error(yMat,linRegressor.predict(XMat)))}")
print(f"MAE with Linear Regression: {mean_absolute_error(yMat,linRegressor.predict(XMat))}")
print(f"MAPE with Linear Regression: {mean_absolute_percentage_error(yMat,linRegressor.predict(XMat))}\n")

plt.show()

#Poly Reg Plot
plt.xticks(np.arange(0, 131, step=5))
x = plt.plot(tenDaysAhead,linRegressor2.predict(polyRegressor.fit_transform(tenDaysAhead)),color="blue")
plt.scatter(X,y,color="red",marker=".")
plt.title("Growth of COVID-19 through 22nd January, 2020 in India (Polynomial, Degree 17)")
plt.xlabel("Days since 22nd January, 2020")
plt.ylabel("Total confirmed cases")

print(f"MSE with Polynomial Regression: {mean_squared_error(yMat,linRegressor2.predict(polyRegressor.fit_transform(XMat)))}")
print(f"RMSE with Polynomial Regression: {sqrt(mean_squared_error(yMat,linRegressor2.predict(polyRegressor.fit_transform(XMat))))}")
print(f"MAE with Polynomial Regression: {mean_absolute_error(yMat,linRegressor2.predict(polyRegressor.fit_transform(XMat)))}")
print(f"MAPE with Polynomial Regression: {mean_absolute_percentage_error(yMat,linRegressor2.predict(polyRegressor.fit_transform(XMat)))}")



plt.show()
