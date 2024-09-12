import numpy as np
import pandas as pd
from matplotlib import pyplot as pt


data = pd.read_csv(train_path)
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)
datahead = data[0:1000].T
head_y = datahead[0]
head_x = datahead[1:n] / 255.
datatrain = data[1000:m].T
train_y = datatrain[0]
train_x = datatrain[1:n] / 255.

def softmax(z):
    return np.exp(z) / sum(np.exp(z))
def wieghtbiasGen():
    w1 = np.random.rand(28,784) - 0.5
    b1 = np.random.rand(28,1) -0.5
    w2 = np.random.rand(10,28) - 0.5
    b2 = np.random.rand(10,1) - 0.5
    return w1,b1,w2,b2

def forward(w1,b1,w2,b2,x):
    z1 = w1.dot(x) + b1
    a1 = np.maximum(0,z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1,a1,z2,a2

def relU_p(z):
    return z > 0

def onehot(y):
    Y = np.zeros((y.size , y.max()+1))
    Y[np.arange(y.size), y] = 1
    Y = Y.T
    return Y

def backprop(z1,a1,z2,a2,w2,x,y):
    Y = onehot(y)
    dz2 = a2 - Y
    dw2 = 1/m*dz2.dot(a1.T)
    db2 = 1/m*np.sum(dz2)
    dz1 = w2.T.dot(dz2) * relU_p(z1)
    dw1 = 1/m*dz1.dot(x.T)
    db1 = 1/m* np.sum(dz1)
    return dw1,db1,dw2,db2,dz2

def upd(w1,b1,w2,b2,alpha,dw1,db1,dw2,db2):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha *db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1,b1,w2,b2

def grad_decend(x,y,iterations, alpha):
    w1 , b1 , w2 , b2 = wieghtbiasGen()
    cycle = iterations // 10
    for i in range(iterations):
        z1 , a1 , z2 , a2 = forward(w1,b1,w2,b2,x)
        dw1 , db1 , dw2 , db2, dz2 = backprop(z1,a1,z2,a2,w2,x,y)
        w1 , b1 , w2 , b2 = upd(w1,b1,w2,b2,alpha,dw1,db1,dw2,db2)
        prediction = np.argmax(a2, 0)
        train_accuracy = np.sum(prediction == y) / y.size
        if train_accuracy > 0.98:
            break
        if i % cycle == 0:
            print(f'iteration number {i} done, model accuracy equals to {train_accuracy}%')
    return w1 , b1 , w2 , b2, train_accuracy

def test_run(x, y, w1, b1, w2, b2):
    z1, a1, z2, a2 = forward(w1, b1, w2, b2, x)
    prediction = np.argmax(a2, 0)
    test_accuracy = np.sum(prediction == y) / y.size
    return test_accuracy
    

iterations, alpha = 30, 0.1
w1, b1, w2, b2, train_accuracy = grad_decend(train_x, train_y, iterations, alpha)
w1d, b1d, w2d, b2d = pd.DataFrame(w1), pd.DataFrame(b1), pd.DataFrame(w2), pd.DataFrame(b2)
w1d.to_csv('data/weights_and_biases/w1.csv')
b1d.to_csv('data/weights_and_biases/b1.csv')
w2d.to_csv('data/weights_and_biases/w2.csv')
b2d.to_csv('data/weights_and_biases/b2.csv')
print(f'train finished succecfully with final model accuracy of {train_accuracy}%, weights and biases are saved in data/weights_and_biases')
test_accuracy = test_run(head_x, head_y, w1, b1, w2, b2)
print(f'test finished with accuracy of {test_accuracy}%')
