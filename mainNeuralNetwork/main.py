import numpy as np
import pandas as pd
data=pd.read_csv("data/train.csv")

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
y_dev = data_dev[0]
x_dev = data_dev[1:n]
x_dev = x_dev / 255.0

data_train = data[1000:m].T
y_train = data_train[0]
x_train = data_train[1:n]
x_train = x_train / 255.0

def param():
    w1=np.random.rand(10,784)-0.5
    b1=np.random.rand(10,1)-0.5
    w2=np.random.rand(10,10)-0.5
    b2=np.random.rand(10,1)-0.5
    return w1,b1,w2,b2

def ReLU(z):
    return np.maximum(0,z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def forwardP(w1,b1,w2,b2,x):
    
    z1=w1.dot(x)+b1
    a1=ReLU(z1)
    z2=w2.dot(a1)+b2
    a2=softmax(z2)
    return z1,a1,z2,a2
def oney(y):
    y=y.astype(int)
    oneY=np.zeros((y.size,10))
    oneY[np.arange(y.size),y] =1
    return oneY.T
 
def deri_relu(z):
    return z>0

def backP(z1, a1, z2, a2, w2, x, y):
    m = y.size
    onehot = oney(y)

    dz2 = a2 - onehot
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)

    dz1 = w2.T.dot(dz2) * deri_relu(z1)
    dw1 = 1 / m * dz1.dot(x.T)
    db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)

    return dw1, db1, dw2, db2

def update_param(dw1,db1,dw2,db2,alpha,w1,b1,w2,b2):
    w1=w1-alpha*dw1
    w2=w2-alpha*dw2
    b1=b1-alpha*db1
    b2=b2-alpha*db2
    return w1,b1,w2,b2

def get_predictions(a2):
    return np.argmax(a2,0)
def get_accuracy(pred,y):
    print(pred,y)
    return np.sum(pred==y)/y.size 

def grad(x,y,iterations,alpha):
    w1,b1,w2,b2=param()
    for i in range(iterations):
        z1,a1,z2,a2=forwardP(w1,b1,w2,b2,x)
        dw1,db1,dw2,db2=backP(z1,a1,z2,a2,w2,x,y)
        w1,b1,w2,b2=update_param(dw1,db1,dw2,db2,alpha,w1,b1,w2,b2)
        if i % 10==0:
            print("iteration: ",i)
            print("Accuracy: ",get_accuracy(get_predictions(a2), y))
    return w1,b1,w2,b2


w1, b1, w2, b2 = grad(x_train, y_train, 1010, 0.1)

np.savez(r"data\mnist.npz", w1=w1, b1=b1, w2=w2, b2=b2)

print("Model saved as mnist.npz")

params = np.load(r"data\mnist.npz")


print(params.files)
print(params["w1"].shape)
print(params["b1"].shape)
print(params["w2"].shape)
print(params["b2"].shape)
