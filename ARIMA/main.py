import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from stationary import *
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import math

#Data Processing
dataset = pd.read_csv(r"D:\Work\NITD\ML\Dataset\total_cases1.csv")
india_cases = dataset["India"]

dates = []
for _ in range(len(india_cases)):
    dates.append(datetime.date(2019,12,30) + datetime.timedelta(_))
    
d={"dates":dates,"india":india_cases}
df = pd.DataFrame(d)
df['india'].fillna(method='ffill',inplace=True)

act_pred = [9.069433214458132861e+05,
            9.364987514580863062e+05,
            9.672083298888972495e+05,
            9.983586209853852633e+05,
            1.030274195702090394e+06,
            1.062502075240437407e+06,
            1.094764580626144540e+06,
            1.127348183517989470e+06,
            1.160639901184762828e+06,
            1.194323885827631457e+06]

for _ in range(len(act_pred)):
    act_pred[_] = int(act_pred[_])

#Stationarity Check
india_diff = df['india'].diff(periods=1)[1:]
plot_acf(india_diff)
plot_pacf(india_diff)

india_diff_diff = india_diff.diff(periods=1)[1:]
plot_acf(india_diff_diff)
plot_pacf(india_diff_diff)

test_stationarity(india_diff)

#Hyper Parameter Optimization
import itertools
import warnings
warnings.filterwarnings('ignore')
p=d=q=range(0,8)
pdq = list(itertools.product(p,d,q))
rmse = []
parameter = 0
for param in pdq:
    try:
        model_india = ARIMA(df["india"][0:185], order=param)
        model_india_fit = model_india.fit()
        india_predictions = model_india_fit.forecast(steps=11)[0]
        rmse.append(math.sqrt(mean_squared_error(df["india"][185:],india_predictions)))
        if rmse[-1] == min(rmse):
            parameter = param
    except:
        continue

#ARIMA
arima_covid = ARIMA(df["india"],(6,2,0)).fit()
#print(arima_covid.summary())

prediction = arima_covid.forecast(steps = 10)[0]
print(f"The RMSE is: {math.sqrt(mean_squared_error(act_pred,prediction.tolist()))}")



