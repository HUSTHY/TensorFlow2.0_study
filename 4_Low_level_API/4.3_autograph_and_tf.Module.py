"""
Autograph的编码规范告诉我们要避免在@tf.function修饰的函数内部定义tf.Variable
但是如果在函数外部定义tf.Variable的话，又会显得函数有外部变量依赖，封装不够好
解决思路：
1、创建一个类，类的初始化方法中定义tf.Variable
2、@tf.function修饰的函数逻辑放在其他的方法中
恰好在TensorFlow中就提供了这样的一个基类
"""

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# x = tf.Variable(2.0,dtype=tf.float32)
# @tf.function(input_signature=[tf.TensorSpec(shape = [], dtype = tf.float32)])
# def add_fun(a):
#     x.assign_add(a)
#     tf.print(x)
#     return x
# add_fun(tf.constant(5.0))
# add_fun(tf.constant(5.2))
#

class DemoModule(tf.Module):
    def __init__(self,init_value = tf.constant(0.0),name=None):
        super(DemoModule,self).__init__(name=name)
        with self.name_scope:
            self.x = tf.Variable(init_value,dtype=tf.float32,trainable=True)


    @tf.function(input_signature=[tf.TensorSpec(shape=[],dtype=tf.float32)])
    def add_fun(self,a):
        with self.name_scope:
            self.x.assign_add(a)
            tf.print('self.x',self.x)
            return self.x


demo = DemoModule(init_value=tf.constant(2.55))
result = demo.add_fun(tf.constant(2.21))
tf.print('result',result)
result = demo.add_fun(tf.constant(3.22))
tf.print('result',result)
print(demo.variables)
print(demo.trainable_variables)
result = demo.add_fun(tf.constant(1.25))

tf.saved_model.save(demo,'saved_model/demo')
demo2 = tf.saved_model.load('saved_model/demo')
demo2.add_fun(tf.constant(100.55))
