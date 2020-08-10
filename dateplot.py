import matplotlib.pyplot as plt
import datetime

def dateplot(X,y,y1):
    dates = []
    for i in range(len(y)):
        dates.append(datetime.date(2020,1,22) + datetime.timedelta(i))
    
    dates10ahead = []
    for i in range(len(y)+10):
        dates10ahead.append(datetime.date(2020,1,22) + datetime.timedelta(i))
    
    plt.scatter(dates,y, c="red", marker=".")
    plt.plot(dates10ahead,y1,c="blue")
    plt.xlabel("Dates")
    plt.ylabel("Total confirmed cases")
    plt.title("Growth of cases of COVID-19 In India")
    plt.show()