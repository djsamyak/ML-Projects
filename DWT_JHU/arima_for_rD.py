import math
import datetime
from stationary import *
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


def arima_rD(rD,rA,df):
    #Data Processing
    dates = []
    for _ in range(len(rD)):
        dates.append(datetime.date(2020,1,30) + datetime.timedelta(_))
        
    d1={"dates":dates,"detailed":rD}
    df1 = pd.DataFrame(d1)
    df1['detailed'].fillna(method='ffill',inplace=True)
    
    #Stationarity Check
    india_diff = df1['detailed'].diff(periods=1)[1:]
    # plot_acf(india_diff)
    # plot_pacf(india_diff)
    
    india_diff_diff = india_diff.diff(periods=1)[1:]
    # plot_acf(india_diff_diff)
    # plot_pacf(india_diff_diff)
    
    # test_stationarity(india_diff_diff)
    
    #Hyper Parameter Optimization
    import itertools
    import warnings
    warnings.filterwarnings('ignore')
    p=d=q=range(0,4)
    pdq = list(itertools.product(p,d,q))
    rmse = []
    parameter = 0
    for param in pdq:
        try:
            model_india = ARIMA(df1["detailed"][:96], order=param)
            model_india_fit = model_india.fit()
            india_predictions = model_india_fit.forecast(steps=10)[0]
            rmse.append(math.sqrt(mean_squared_error(df1["detailed"][96:],india_predictions)))
            if rmse[-1] == min(rmse):
                parameter = param
        except:
            continue
    
    #ARIMA
    arima_values = ARIMA(df1["detailed"][:96],(3,2,0)).fit()
    #print(arima_values.summary())
    
    prediction = arima_values.forecast(steps = 10)[0]
    residual_arima = df1["detailed"][96:] - prediction
    
    dataset = []
    for _ in range(len(rA)):
        if _<96:
            dataset.append(rA[_])
        else:
            dataset.append(rA[_] + residual_arima[_])
        
    print(f"The RMSE is: {math.sqrt(mean_squared_error(df['india'][103:113].tolist(),dataset[96:]))}")
    
    return residual_arima,dataset,df1



