"""
主要是看看@tf.function装饰器起的作用
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

@tf.function
def myfun(a,b):
    for i in tf.range(5):
        tf.print('i',i)
    c = a+b
    print('tracing')
    return c
myfun(tf.constant('hello'),tf.constant('tensorflow'))#创建计算图，执行print('tracing')，然后执行计算图
myfun(tf.constant('NLP'),tf.constant('study'))#不需要创建计算图，就不用执行print('tracing')，只需要执行计算图
myfun(tf.constant('NLP'),tf.constant('study'))#不需要创建计算图，就不用执行print('tracing')，只需要执行计算图
myfun(tf.constant(1),tf.constant(2))#由于参数类型不同，原来创建的计算图不适用，所以要重新创建计算图，就需要执行print('tracing')
myfun(1,2)#@tf.function装饰的函数时输入的参数不是Tensor类型，则每次都会重新创建计算图
myfun(3,4)#@tf.function装饰的函数时输入的参数不是Tensor类型，则每次都会重新创建计算图

