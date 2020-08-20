import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras import models,layers,optimizers


class MyModel(models.Model):
    def __init__(self,max_word,cat_num):
        self.max_word = max_word
        self.cat_num = cat_num
        super(MyModel,self).__init__()

    def build(self,input_shape):
        self.emb = layers.Embedding(self.max_word,7,input_length=300)
        self.conv1 = layers.Conv1D(filters=64,kernel_size=5,activation='relu')
        self.conv2 = layers.Conv1D(filters=32,kernel_size=3,activation='relu')
        self.pool = layers.MaxPool1D(2)
        self.flat = layers.Flatten()
        self.clas = layers.Dense(self.cat_num,activation='softmax')
        super(MyModel,self).build(input_shape)

    def call(self, inputs):
        x = self.emb(inputs)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        f = self.flat(x)
        out_put = self.clas(f)
        return out_put

@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features,training=True)
        loss = loss_func(labels,predictions)
    gradients = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))

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
        for  features, labels in ds_train:
            train_step(model,features, labels)
        for features, labels in ds_valid:
            valid_step(model,features,labels)

        logs = 'Epoch={},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{}'
        if epoch%1 == 0:
            printbar()
            tf.print(tf.strings.format(logs,(epoch,train_loss.result(),train_metric.result(),valid_loss.result(),valid_metric.result())))
            tf.print("")

        train_loss.reset_states()
        train_metric.reset_states()
        valid_loss.reset_states()
        valid_metric.reset_states()


@tf.function
def printbar():
    today = tf.timestamp()%(24*60*60)

    hour = tf.cast(today//3600+8,tf.int32)%tf.constant(24)
    minite = tf.cast((today%3600)//60,tf.int32)
    second = tf.cast(tf.floor(today%60),tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}", m)) == 1:
            return (tf.strings.format("0{}", m))
        else:
            return (tf.strings.format("{}", m))

    timestrins = tf.strings.join([timeformat(hour),timeformat(minite),
                timeformat(second)],separator = ":")
    tf.print('===='*20+timestrins)

if __name__ == '__main__':
    MAX_LEN = 300
    BATCH_SIZE = 64
    (x_train, y_train), (x_test, y_test) = datasets.reuters.load_data()
    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_LEN)

    MAX_WORDS = x_train.max() + 1
    CAT_NUM = y_train.max() + 1

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .shuffle(buffer_size=100).batch(BATCH_SIZE) \
        .prefetch(tf.data.experimental.AUTOTUNE).cache()

    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
        .shuffle(buffer_size=100).batch(BATCH_SIZE) \
        .prefetch(tf.data.experimental.AUTOTUNE).cache()

    lr = 0.001
    optimizer = optimizers.Nadam()
    loss_func = losses.SparseCategoricalCrossentropy()
    
    train_loss = metrics.Mean(name='train_loss')
    train_metric = metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = metrics.Mean(name='valid_loss')
    valid_metric = metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    model = MyModel(MAX_WORDS,CAT_NUM)
    model.build(input_shape=(None, MAX_LEN))
    model.summary()

    train_model(model,ds_train,ds_test,100)
    
