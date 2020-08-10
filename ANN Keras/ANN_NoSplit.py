import math
import datetime
import numpy as np
import pandas as pd
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#Seed Stabilization
seed_value = 20
import tensorflow as tf
import random
import numpy as np
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

#Data Processing
dataset = pd.read_csv(r"D:\Work\NITD\ML\Dataset\total_cases1.csv")
dates = []
X=[]
for _ in range(len(dataset)):
    dates.append(datetime.date(2019,12,30) + datetime.timedelta(_))
for _ in range(len(dataset)):
    X.append(_)
    
d={"X":X,"dates":dates,"india":dataset["India"]}
df = pd.DataFrame(d)
df['india'].fillna(method='ffill',inplace=True)

#Model Creation
model = Sequential()
model.add(Dense(30,activation='relu',use_bias=True,kernel_initializer ='he_uniform',input_shape=(1,)))
model.add(Dense(15,activation='relu',use_bias=True,kernel_initializer ='he_uniform'))
model.add(Dense(1))
opt = Adam(learning_rate=0.01)
model.compile(opt,"mean_squared_error",["mean_absolute_error"])
model.fit(df["X"],df["india"],3,600)
prediction = model.predict(df["X"])
model.summary()

#Plotting
print(f"The RMSE error with the prediction is: {math.sqrt(mean_squared_error(df['india'],prediction))}")
plt.scatter(df["X"],df["india"],color="red")

prediction_list = prediction.tolist()
for i in range(len(prediction_list)):
    prediction_list[i] = prediction_list[i][0]
    
plt.scatter(df["X"],prediction,color="blue",marker=".")
plt.show()

# def NARNN(delay): #use for loop and add each element
#     y=0
#     #y=model.predict((df["X"]-delay)[delay:])
#     for i in delay:
#         y += 
#     y=model.predict((df["X"]-delay)[delay:]) + model.predict((df["X"]-delay-1)[delay-1:]) 
#     return y

# NARNN_prediction = NARNN(1)