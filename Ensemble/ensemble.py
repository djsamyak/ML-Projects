import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Polynomial Regression")
from pr import *

print("\nExponential Smoothing")
from es import *

# print("\nSVR")
# from svr import * 

print("\nANN")
from ANN_scratch import *

#PR,ES,SVR
weights = [1.88,1.1,0.02]
# prediction_models = [ydata,yhat1,predited_values1] for SVR
prediction_models = [ydata,yhat1,predicted_ahead]

dataset = pd.read_csv(r"C:\Users\djsam\Desktop\ML\Dataset\covid19.csv")
y_values = dataset.iloc[131:132,4:].values

ensemble_values = []
for i in np.arange(0,3):
    ensemble_values.append((weights[i]*prediction_models[i]))

final_predicted_value = sum(ensemble_values)/3

print("\nEnsemble  Model")    
print(f"RMSE = {mean_squared_error(y,final_predicted_value[:176],squared = False)}")
print(f"MSE = {mean_squared_error(y,final_predicted_value[:176],squared = True)}")
print(f"MAE = {mean_absolute_error(y,final_predicted_value[:176])}")
print(f"MAPE = {mean_absolute_percentage_error(y,final_predicted_value[:176])} ")

plt.scatter(dates,y_values, c="black", marker=".", label="True data")
plt.plot(dates10ahead,predicted_ahead,c="red",label="ANN")
plt.plot(dates10ahead,ydata,c="blue",label="Polynomial Regression (Degree 11)")
plt.plot(dates10ahead,yhat1,c="green",label="Exponential Smoothing")
plt.plot(dates10ahead,final_predicted_value,c="magenta",label="Average Weighted Ensemble Model")

plt.legend()
plt.xlabel("Months")
plt.ylabel("Total confirmed cases")
plt.title("Growth of cases of COVID-19 In India")
plt.show()


