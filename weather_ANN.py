import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense

dataset = pd.read_csv(r"C:\Users\djsam\Desktop\ML\Dataset\2105944.csv")
all_stations= dataset.iloc[:,1]

stations = ['BHUBANESWAR, IN']
i=0
counter=0
while(i<len(dataset)):
    if (all_stations[i] not in stations):
        stations.append(all_stations[i])
    i+=1

i=0
station_frequency = {}
while(i<len(stations)):
    station_frequency.update({stations[i]:0})
    i=i+1

i=0
while(i<len(dataset)):
    if (all_stations[i] in stations):
        station_frequency[all_stations[i]]+=1
    i+=1   

#End of data pre-processing
############################################################################################

i=0
while(i<len(stations)):
    print(f"{i+1}. {stations[i]}")
    i+=1
print("\nWhich place's temperature would you like to predict?")
userChoice = int(input()) - 1
selectedStation = stations[userChoice]
print(f"Selected Station: {selectedStation}")
if(station_frequency[selectedStation]<25):
    print("NOT ENOUGH DATA")

i=0
while(i<len(all_stations)):
    if(all_stations[i] == selectedStation):
        startIndex = i
        break
    i=i+1

endIndex = station_frequency[selectedStation] + startIndex
selectedStationData = dataset.iloc[startIndex:endIndex,:]

dates = selectedStationData.iloc[:,5:6].values
mean_temperature = selectedStationData.iloc[:,8:9].values

#End of obtaining data
############################################################################################

model = Sequential()
model.add(Dense(10,input_shape=(1,),activation='linear'))
model.add(Dense(10,activation='sigmoid'))
model.add(Dense(10,activation='linear'))
model.add(Dense(10,activation='sigmoid'))
model.add(Dense(10,activation='linear'))
model.add(Dense(1))

numericDates = np.arange(0,len(dates))

model.compile("adam","mean_squared_error",["mean_absolute_error"])
model.fit(numericDates,mean_temperature,2,2000)

prediction = model.predict(numericDates)
model.summary()
plt.scatter(numericDates,mean_temperature,color="red",marker=".")
plt.plot(numericDates,prediction,color="blue")
plt.title("Temperature changes")
plt.xlabel("Days since 1st February, 2020")
plt.ylabel("Temperature in Fahrenheit")
plt.show()






