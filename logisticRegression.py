import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# %matplotlib inline

x = pd.read_csv("Logistic_X_Train.csv")
y = pd.read_csv("Logistic_Y_Train.csv")
x = x.values
y = y.values
y_ = y.reshape((-1,1))
print(x.shape,y.shape,type(x),type(y_))
ones = np.ones((x.shape[0],1))
x_ = np.hstack((ones,x))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def hypothesis(x,theta):
    return sigmoid(np.dot(x,theta))

def error(x,y,theta):
    hyptesis = hypothesis(x,theta)
    err = y*np.log(hyptesis) + (1-y)*np.log(1-hyptesis)
    return -1*np.mean(err)

def gradient(x,y,theta):
    hi = hypothesis(x,theta)
    grad =  np.dot(x.T,(y-hi))
    return grad/x.shape[0]

def gradient_descent(x,y,lr=1,itr=1000):
    theta = np.zeros((x.shape[1],1))
    error_list = []

    for i in range(itr):
        e = error(x,y,theta)
        error_list.append(e)

        grad = gradient(x,y,theta)
        theta = theta + lr*grad
    
    return theta,error_list

theta,error_list = gradient_descent(x_,y_)
print(theta)
# plt.scatter(x_[:,1],x_[:,2],c=y_.reshape((-1,)),cmap=plt.cm.Accent)

# plt.plot(x_axis,y_axis)
# plt.show()
# plt.style.use('seaborn')
# plt.plot(error_list)
# plt.show()

def predict(x,theta):
    # h - m*1
    h = hypothesis(x,theta)
    output = np.zeros(h.shape)

    output[h>=0.5] = 1
    output = output.astype('int')
    # print(output)
    return output

preds = predict(x_,theta)

def accuracy(actual,preds):
    actual = actual.astype('int')
    print(np.sum(actual==preds))

accuracy(y_,preds)

x_axis = np.arange(-3,4)
y_axis = -(theta[0]+theta[1]*x_axis+theta[2]*x_axis)/theta[3]
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x_[:,1],x_[:,2],x_[:,3])
plt.plot(x_axis,y_axis,color='red')
plt.show()

# test = pd.read_csv("Logistic_X_Test.csv")
# test = test.values
# ones = np.ones((test.shape[0],1))
# test_ = np.hstack((ones,test))
# preds = predict(test_,theta)

# print(type(preds))

# df = pd.DataFrame(preds,columns=['label'])
# print(df)
# df.to_csv("answer.csv",index=False)