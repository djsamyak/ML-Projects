import numpy as np
from sklearn.metrics import mean_squared_error

def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100

def activation(X):
    i=0
    while(i < len(X[0,:])):
        if X[0,i] > 0:
            pass
        else:
            X[0,i] = 0
        i=i+1
    return X

def activation_derivative(X):
    i=0
    abc = []
    while(i < len(X[0,:])):
        if X[0,i] > 0:
            abc.append(1)
        else:
            abc.append(0)
        i=i+1
    xabc = np.array(abc)
    xabc = np.reshape(xabc,(1,-1))
    return xabc

def Z_calc(theta,bias,X,i):
    z=0 
    z = np.dot(theta[i,0],np.transpose(X)) + bias
    return z

def loss_function(y, a_output):
    m = len(y[0,:])
    #J = (-1/m)*np.sum(y*np.log(a_output) + (1-y)*(np.log(1-a_output)))
    J = mean_squared_error(y,a_output,squared = False)
    
    return J

def forward_prop(X,y,theta1,theta2,bias1,bias2,HIDDEN_LAYER_1_NODES):
    a={}
    i=0
    z1=0
    
    while (i<HIDDEN_LAYER_1_NODES):
        a[i] = activation(Z_calc(theta1,bias1,X,i))
        z1 += Z_calc(theta1,bias1,X,i)
        i=i+1
    
    i=0
    a_hidden=0
    while (i<HIDDEN_LAYER_1_NODES):
        a_hidden += a[i]
        i+=1
    
    i=0
    z_layer2=0
    while (i<HIDDEN_LAYER_1_NODES):
        z_layer2 += np.dot(theta2[0,i],a[i]) + bias2
        i+=1
    a_output = activation(z_layer2)
    a_output = np.transpose(a_output)

    i=0
    while(i<len(a_output)):
        if (a_output[i,0] == 0):
            a_output[i,0]=0.00000000001
        i=i+1
    
    m = X.shape[0]
    J = loss_function(y,a_output)
    
    dz2 = a_output - y
    dw2 = (1/m) * np.dot(np.transpose(dz2), np.transpose(a_hidden))
    db2 = (1/m) * np.sum(dz2)
    
    dz1 = (np.transpose(theta2)*np.transpose(dz2)) @ np.transpose(activation_derivative(z1))
    dw1 = (1/m) * dz1 * np.transpose(X)
    db1 = (1/m) * np.sum(dz1)
    
    grads = {"dz2":dz2,
             "dw2":dw2,
             "db2":db2,
             "dz1":dz1,
             "dw1":dw1,
             "db1":db1}
    
    return J,grads,a_output


def back_prop(X,y,theta1,theta2,bias1,bias2,num_iterations, learning_rate, HIDDEN_LAYER_1_NODES, print_cost = False):
    costs = []
    prediction_history = []
    
    for iter123 in range(num_iterations):
        if iter123>20000:
            learning_rate = 0.000003
        elif iter123>35000:
            learning_rate = 0.000007
        
        J,grads,a_output = forward_prop(X,y,theta1,theta2,bias1,bias2,HIDDEN_LAYER_1_NODES)
        
        theta1 = theta1 - learning_rate*grads["dw1"]
        theta2 = theta2 - learning_rate*grads["dw2"]
        bias1 = bias1 - learning_rate*grads["db1"]
        bias2 = bias2 - learning_rate*grads["db2"]
        
        if iter123 % 1000 == 0:
            costs.append(J)
        
        #print(iter123)

        if print_cost and iter123 % 1000 == 0:
            print ("Cost after iteration %i: %f" %(iter123, J ))
        
        if iter123 < 100:
            prediction_history.append(a_output)
            
        X_ahead = []
        for i in range(len(X)+10):
            X_ahead.append(i)
        X_ahead = np.array(X_ahead)
        X_ahead = np.reshape(X_ahead,(-1, 1))
        
        a={}
        i=0
        z1=0
        
        while (i<HIDDEN_LAYER_1_NODES):
            a[i] = activation(Z_calc(theta1,bias1,X_ahead,i))
            z1 += Z_calc(theta1,bias1,X_ahead,i)
            i=i+1
        
        i=0
        a_hidden=0
        while (i<HIDDEN_LAYER_1_NODES):
            a_hidden += a[i]
            i+=1
        
        i=0
        z_layer2=0
        while (i<HIDDEN_LAYER_1_NODES):
            z_layer2 += np.dot(theta2[0,i],a[i]) + bias2
            i+=1
        predicted_ahead = activation(z_layer2)
        predicted_ahead = np.transpose(predicted_ahead)

    return a_output,prediction_history,predicted_ahead
        
        
        
        
        
        
        
        
        
        
        
        
        
        