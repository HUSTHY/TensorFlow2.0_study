"""
这里一般都是组合了很多基础模型层
Dense：密集连接层
Activation：激活函数层
Dropout：随机置零层
Conv1D：普通一维卷积
Conv2D：普通二维卷积
MaxPool2D: 二维最大池化层
LSTM：长短记忆循环网络层
Attention：Dot-product类型注意力机制层
"""

"""
如果需要定义自己的模型层可以通过对Layer基类子类化实现
Layer的子类化一般需要重新实现初始化方法，Build方法和Call方法
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras import models,layers

class Linear_Layer(layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(Linear_Layer, self).__init__(**kwargs)
        self.units = units


    def build(self, input_shape):
        self.w = self.add_weight("w",shape=(input_shape[-1], self.units),#注意必须要有参数名称"w",否则会报错
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight("b",shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)
        super(Linear_Layer,self).build(input_shape)# 相当于设置self.built = True


    @tf.function
    def call(self, inputs, **kwargs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(Linear_Layer,self).get_config()
        config.update({'units': self.units})
        return config

linear1 = Linear_Layer(units=16)
input = tf.random.uniform((100,64))
tf.print(input)
result  = linear1(input)
tf.print(result.shape)
config = linear1.get_config()
print(config)
print('*'*100)

class Linear_model(models.Model):
    def __init__(self,units=32):
        super(Linear_model,self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight("w",shape=(input_shape[-1], self.units),#注意必须要有参数名称"w",否则会报错
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight("b",shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)
        super(Linear_model,self).build(input_shape)

    @tf.function
    def call(self, inputs, **kwargs):
        return tf.matmul(inputs, self.w) + self.b



linear1 = Linear_model(units=10)
input = tf.random.uniform((100,64))
tf.print(input)
result  = linear1(input)
tf.print(result.shape)
linear1.summary()
