from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from ANN_scratch import *

dataset = pd.read_csv(r"C:\Users\djsam\Desktop\ML\COVID19_Data\COVID-19\csse_covid_19_data\csse_covid_19_time_series\time_series_covid19_confirmed_global_new1.csv")
X = list(dataset.columns)
X = X[4:]                               #days
y = dataset.iloc[131:132,4:].values     #Confimed Patients

X = np.array(X)
X = X.astype(int)
X = np.reshape(X,(-1, 1))
y = np.reshape(y,(-1, 1))               #Converted y to appropriate form  - y.transpose()
y_scaled = (y)/y[-1,0]

gsc = GridSearchCV(estimator = model(X,y,0.0000001,15),
                   param_grid = { 'learning_rate': [0.0000001,0.0000003,0.0000009,0.000001,0.000003,0.000009,0.00001],
                                  'HIDDEN_LAYER_1_NODES': [10,12,14,16,17,18,20]},
                   cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

grid_result = gsc.fit(X,y.ravel())
best_params = grid_result.best_params_