from sklearn.model_selection import train_test_split
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

def ann_combined(rA,residual_arima):
    #Data Processing
    dataset = []
    dates = []
    X=[]
    for _ in range(len(rA)):
        if _<98:
            dataset.append(rA[_])
        else:
            dataset.append(rA[_] + residual_arima[_])
    
    for _ in range(len(dataset)):
        dates.append(datetime.date(2020,1,30) + datetime.timedelta(_))
    for _ in range(len(dataset)):
        X.append(_)
        
    d2={"dates":dates,"X":X,"dataset":dataset}
    df2 = pd.DataFrame(d2)
    
    X_train,X_test,y_train,y_test = train_test_split(df2["X"],df2["dataset"],test_size=0.10185,shuffle=False)
    
    #Model Creation
    model = Sequential()
    model.add(Dense(30,activation='relu',use_bias=True,kernel_initializer ='he_uniform',input_shape=(1,)))
    model.add(Dense(22,activation='relu',use_bias=True,kernel_initializer ='he_uniform'))
    model.add(Dense(1))
    
    opt = Adam(learning_rate=0.3)
    model.compile(opt,"mean_squared_error",["mean_absolute_error"])
    model.fit(X_train,y_train,5,2000)
    prediction = model.predict(X_train)
    
    prediction1 = model.predict(X_test)
    model.summary()
    
    #prediction2 = model.predict(X)
    
    print(f"The RMSE error with the prediction is: {math.sqrt(mean_squared_error(y_test,prediction1))}")

    return prediction,prediction1,df2