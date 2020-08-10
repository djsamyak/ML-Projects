import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scratch_functions import *
import datetime

dataset = pd.read_csv(r"C:\Users\djsam\Desktop\ML\COVID19_Data\COVID-19\csse_covid_19_data\csse_covid_19_time_series\time_series_covid19_confirmed_global_new1.csv")
X = list(dataset.columns)
X = X[4:]                               #days
y = dataset.iloc[131:132,4:].values     #Confimed Patients

X = np.array(X)
X = X.astype(int)
X = np.reshape(X,(-1, 1))
y = np.reshape(y,(-1, 1))               #Converted y to appropriate form  - y.transpose()

#Hyper-Parameters
INPUT_LAYER_NODES = 1
HIDDEN_LAYER_1_NODES = 10
OUTPUT_LAYER_NODES = 1
np.random.seed(73)
theta1  = np.random.rand(HIDDEN_LAYER_1_NODES,INPUT_LAYER_NODES )
theta2  = np.random.rand(OUTPUT_LAYER_NODES,HIDDEN_LAYER_1_NODES)
bias1 = 0
bias2 = 0
learning_rate = 0.00000007

predicted_value,history,predicted_ahead = back_prop(X, y, theta1, theta2, bias1, bias2, 50000, learning_rate, HIDDEN_LAYER_1_NODES, print_cost = True)

#gsc = GridSearchCV(estimator = GradientBoostingRegressor(),
#                   param_grid = { 'learning_rate': [0.000001,0.000003,0.000009,0.00001,0.00003,0.00009,0.0001,0.0003,0.0009,0.001,0.003,0.009,0.01,0.03,0.09]},
#                   cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
#
#grid_result = gsc.fit(X,y.ravel())
#best_params = grid_result.best_params_

print(f"\nMSE = {mean_squared_error(y,predicted_value,squared = False)}")
print(f"RMSE = {mean_squared_error(y,predicted_value,squared = True)}")
print(f"MAE = {mean_absolute_error(y,predicted_value)}")
print(f"MAPE = {mean_absolute_percentage_error(y,predicted_value)}")

# pd.plotting.autocorrelation_plot(y)
# plt.show()

# dates = []
# for i in range(len(yhat)):
#     dates.append(datetime.date(2020,1,22) + datetime.timedelta(i))

# plt.scatter(dates,y, c="red", marker=".")
# plt.plot(dates,yhat,c="blue")
# plt.xlabel("Dates")
# plt.ylabel("Total confirmed cases")
# plt.title("Growth of cases of COVID-19 In India")
# plt.show()

dates = []
for i in range(len(X)):
    dates.append(datetime.date(2020,1,22) + datetime.timedelta(i))

dates10ahead = []
for i in range(len(X)+10):
    dates10ahead.append(datetime.date(2020,1,22) + datetime.timedelta(i))

plt.scatter(dates,y, c="red", marker=".")
plt.plot(dates10ahead,predicted_ahead,c="blue")
plt.xlabel("Dates")
plt.ylabel("Total confirmed cases")
plt.title("Growth of cases of COVID-19 In India")
plt.show()




