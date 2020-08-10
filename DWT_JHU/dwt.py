import pywt
import datetime
import numpy as np
import ann_combined 
import arima_for_rD 
import pandas as pd
from numpy.random import seed
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#Data Processing
dataset = pd.read_csv(r"D:\Work\NITD\ML\Dataset\covid19.csv")
india_cases = dataset.iloc[131][4:]

dates = []
for _ in range(len(india_cases)):
    dates.append(datetime.date(2020,1,22) + datetime.timedelta(_))
    
d={"dates":dates,"india":india_cases}
df = pd.DataFrame(d)
df['india'].fillna(method='ffill',inplace=True)

#Decomposition and Recomposition for Detailed Coeffs
coeffs = pywt.wavedec(df['india'][8:115], 'coif10',level=2)
coeffs[0] = np.zeros_like(coeffs[0])
rD = pywt.waverec(coeffs,'coif10')

#Decomposition and Recomposition for Approximate Coeffs
coeffs = pywt.wavedec(df['india'][8:115], 'coif10',level=2)
coeffs[1] = np.zeros_like(coeffs[1])
coeffs[2] = np.zeros_like(coeffs[2])
rA = pywt.waverec(coeffs,'coif10')

# plt.plot(dates,rD)
# plt.show()

residual_arima = arima_for_rD.arima_rD(rD)
prediction2,df2 = ann_combined.ann_combined(rA,residual_arima)

plt.scatter(df2["dates"],df["india"],color="red",label="Observed")
plt.scatter(df2["dates"],prediction2,color="blue",marker=".",label="Predicted")
plt.xlabel("Months")
plt.title("Growth of SARS-CoV-2 in India")
plt.legend(loc="left")
plt.ylabel("Total Cumulative Cases")
plt.show()
