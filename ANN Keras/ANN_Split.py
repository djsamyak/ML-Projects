import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

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

X_train,X_test,y_train,y_test = train_test_split(df["X"],df["india"])

act_pred = "906752 936181 968857 1003832 1039084 1077781 1118206 1155338 1193078 1238798".split(" ")
for _ in range(len(act_pred)):
    act_pred[_] = int(act_pred[_])

#Model Creation
model = Sequential()
model.add(Dense(30,activation='relu',use_bias=True,kernel_initializer ='he_uniform',input_shape=(1,)))
model.add(Dense(15,activation='relu',use_bias=True,kernel_initializer ='he_uniform'))
model.add(Dense(1))
opt = Adam(learning_rate=0.01)
model.compile(opt,"mean_squared_error",["mean_absolute_error"])
model.fit(X_train,y_train,3,700)
prediction = model.predict(X_train)
model.summary()

prediction1 = model.predict(X_test)
model.summary()

print(f"The RMSE error with the prediction is: {math.sqrt(mean_squared_error(y_test,prediction1))}")
plt.scatter(X,df["india"],color="red")
plt.scatter(X_train,prediction,color="blue",marker=".")
plt.scatter(X_test,prediction1,color="blue",marker=".")
plt.show()