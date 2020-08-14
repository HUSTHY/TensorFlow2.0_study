import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras import layers,losses,metrics,optimizers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

@tf.function
def printbar():
    today_ts = tf.timestamp()%(24*60*60)
    hour = tf.cast(today_ts//3600+8,tf.int32)%tf.constant(24)
    minute = tf.cast((today_ts%3600)//60,tf.int32)
    second = tf.cast(tf.floor(today_ts%60),tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format('{}',m))==1:
            return (tf.strings.format('0{}',m))
        else:
            return (tf.strings.format('{}',m))

    timestring = tf.strings.join([timeformat(hour),timeformat(minute),timeformat(second)],separator=':')
    tf.print('==========='*8+timestring)



n = 1000

X = tf.random.uniform([n,2],minval=-10,maxval=10,seed=1)
w0 = tf.constant([[2.0],[-3.0]])
b0 = tf.constant([[3.0]])
Y = X@w0 + b0 +tf.random.normal([n,1],mean = 0.0,stddev= 2.0)# @表示矩阵乘法,增加正态扰动

# 数据可视化
plt.figure(figsize = (12,5))
ax1 = plt.subplot(121)
ax1.scatter(X[:,0],Y[:,0], c = "b")
plt.xlabel("x1")
plt.ylabel("y",rotation = 0)

ax2 = plt.subplot(122)
ax2.scatter(X[:,1],Y[:,0], c = "g")
plt.xlabel("x2")
plt.ylabel("y",rotation = 0)
plt.show()


ds = tf.data.Dataset.from_tensor_slices((X,Y)).shuffle(buffer_size=100).batch(20).prefetch(tf.data.experimental.AUTOTUNE)

model = layers.Dense(units=1)
model.build(input_shape=(2,))
model.loss_func = losses.mean_squared_error
model.optimizer = optimizers.SGD(learning_rate=0.001)

@tf.function
def train_step(model,features,labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = model.loss_func(tf.reshape(labels,[-1]),tf.reshape(predictions,[-1]))
    grads =  tape.gradient(loss,model.variables)
    model.optimizer.apply_gradients(zip(grads,model.variables))
    return loss

def train_model(model,epochs):
    for epoch in tf.range(1,epochs+1):
        for features,labels in ds:
            loss = train_step(model,features,labels)
        if epoch%5 == 0:
            printbar()
            tf.print('epoch=',epoch)
            tf.print('loss=', loss)
            tf.print('w=', model.variables[0])
            tf.print('b=', model.variables[1])

train_model(model,epochs = 400)


w,b = model.variables

plt.figure(figsize = (12,5))
ax1 = plt.subplot(121)
ax1.scatter(X[:,0],Y[:,0], c = "b",label = "samples")
ax1.plot(X[:,0],w[0]*X[:,0]+b[0],"-r",linewidth = 5.0,label = "model")
ax1.legend()
plt.xlabel("x1")
plt.ylabel("y",rotation = 0)



ax2 = plt.subplot(122)
ax2.scatter(X[:,1],Y[:,0], c = "g",label = "samples")
ax2.plot(X[:,1],w[1]*X[:,1]+b[0],"-r",linewidth = 5.0,label = "model")
ax2.legend()
plt.xlabel("x2")
plt.ylabel("y",rotation = 0)

plt.show()