import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv(r"C:\Users\djsam\Desktop\ML\Dataset\covid19.csv")
X = dataset.iloc[131,4:].index
y = dataset.iloc[131,4:].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

x1 = np.array(X) 

regressor = LinearRegression()
regressor.fit(np.reshape(X_train,(-1, 1)),np.reshape(y_train,(-1, 1)))
prediction = regressor.predict(np.reshape(X_test,(-1, 1)))

plt.xticks(np.arange(0, 121, step=10))
plt.scatter(X,y,color="red",marker=".")
plt.plot(X_train,regressor.predict(np.reshape(X_train,(-1, 1))),color="blue")
plt.title("Growth of COVID-19 through 22nd January, 2020 in India")
plt.xlabel("Days since 22nd January, 2020")
plt.ylabel("Total confirmed cases")

plt.show()