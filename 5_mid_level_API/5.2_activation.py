"""
tensorflow中沿用的是keras的风格，模型中使用
激活函数有2中方法
一种是activation参数指定
一种是显示添加layers.Activation
"""
import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras import layers,models

tf.keras.backend.clear_session()

model = models.Sequential()
model.add(layers.Dense(32,input_shape=(None,32),activation=tf.nn.relu))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(16))
model.add(layers.Activation(tf.nn.sigmoid))
model.summary()
