import tensorflow as tf
from tensorflow.keras import models,layers,losses,metrics,optimizers
import numpy as np
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




n_pos,n_nege = 2000,2000
r_p = 5.0 + tf.random.truncated_normal([n_pos,1],0.0,1.0)
theta_p = tf.random.uniform([n_pos,1],0.0,2*np.pi)
xp = tf.concat([r_p*tf.cos(theta_p),r_p*tf.sin(theta_p)],axis=1)
yp = tf.ones_like(r_p)

r_n = 8.0 + tf.random.truncated_normal([n_nege,1],0.0,1.0)
theta_n = tf.random.uniform([n_nege,1],0.0,2*np.pi)
Xn = tf.concat([r_n*tf.cos(theta_n),r_n*tf.sin(theta_n)],axis = 1)
Yn = tf.zeros_like(r_n)

x = tf.concat([xp,Xn],axis = 0)
y = tf.concat([yp,Yn],axis = 0)

data = tf.concat([x,y],axis=1)
data = tf.random.shuffle(data)
x = data[:,:2]
y = data[:,2:]

ds_train = tf.data.Dataset.from_tensor_slices((x[0:3000,:],y[0:3000,:])) .shuffle(buffer_size=500).batch(10).prefetch(tf.data.experimental.AUTOTUNE).cache()
ds_valid = tf.data.Dataset.from_tensor_slices((x[3000:,:],y[3000:,:])).batch(10).prefetch(tf.data.experimental.AUTOTUNE).cache()


tf.keras.backend.clear_session()
class DNNModel(models.Model):
    def __init__(self):
        super(DNNModel,self).__init__()

    def build(self,input_shape):
        self.dense1 = layers.Dense(4,activation='relu',name='dense1')
        self.dense2 = layers.Dense(8,activation='relu',name='dense2')
        self.dense3 = layers.Dense(1,activation='relu',name='dense3')
        super(DNNModel,self).build(input_shape)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        y = self.dense3(x)
        return y


model = DNNModel()
model.build(input_shape=(None,2))
model.summary()

optimizer = optimizers.Adam(learning_rate=0.001)
loss_func = tf.keras.losses.BinaryCrossentropy()

train_loss = tf.keras.metrics.Mean(name='trian_loss')
train_metric = tf.keras.metrics.BinaryAccuracy(name='train_metric')

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_metric = tf.keras.metrics.BinaryAccuracy(name='valid_metric')


@tf.function
def train_step(model,feature,labels):
    with tf.GradientTape() as tape:
        predictions = model(feature)
        loss = loss_func(labels,predictions)
    grads = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    train_loss.update_state(loss)
    train_metric.update_state(labels,predictions)
    
@tf.function
def valid_step(model, features, labels):
    predictions = model(features)
    batch_loss = loss_func(labels,predictions)
    valid_loss.update_state(batch_loss)
    valid_metric.update_state(labels,predictions)


def train_model(model,ds_train,ds_valid,epochs):
    for epoch in tf.range(1,epochs+1):
        for features,labels in ds_train:
            train_step(model,features,labels)
        for features,labels in ds_valid:
            valid_step(model,features,labels)

        logs = 'Epoch={},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{}'
        if epoch % 20 == 0:
            printbar()
            tf.print(tf.strings.format(logs,
            (epoch,train_loss.result(),train_metric.result(),valid_loss.result(),valid_metric.result())))

        train_loss.reset_states()
        valid_loss.reset_states()
        train_metric.reset_states()
        valid_metric.reset_states()

train_model(model,ds_train,ds_valid,2000)