"""
有三种计算图的构建方式：静态计算图，动态计算图，以及Autograph。
ensorFlow 2.0主要使用的是动态计算图和Autograph。
动态计算图易于调试，编码效率较高，但执行效率偏低。
静态计算图执行效率很高，但较难调试。
Autograph机制可以将动态图转换成静态计算图，兼收执行效率和编码效率之利。
"""

"""
"Autograph编码规范总结"
1、被@tf.function修饰的函数应尽可能使用TensorFlow中的函数而不是Python中的其他函数。例如使用tf.print而不是print，使用tf.range而不是range，使用tf.constant(True)而不是True.
2、避免在@tf.function修饰的函数内部定义tf.Variable.
3、被@tf.function修饰的函数不可修改该函数外部的Python列表或字典等数据结构变量。
"""

import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# 1、被@tf.function修饰的函数应尽可能使用TensorFlow中的函数而不是Python中的其他函数。例如使用tf.print而不是print，使用tf.range而不是range，使用tf.constant(True)而不是True.
@tf.function
def np_random():
    a = np.random.randn(3,3)
    tf.print(a)

@tf.function
def tf_random():
    a = tf.random.normal((3,3))
    tf.print(a)

for _ in range(5):
    # np_random运行的每次结果都是一样的，而tf_random每次都会重新生成
    np_random()
    tf_random()

# 2、避免在@tf.function修饰的函数内部定义tf.Variable.
x = tf.Variable(1.0,dtype=tf.float32)

@tf.function
def outer_var():
    x.assign_add(0.55)
    tf.print(x)
    return x

@tf.function
def inner_var():
    x = tf.Variable(3.14,dtype=tf.float32)
    x.assign_add(2.36)
    tf.print(x)
    return x

outer_var()
# inner_var()#执行这个就会报错

# 3、被@tf.function修饰的函数不可修改该函数外部的Python列表或字典等结构类型变量。
tensor_list = []
def append_tensor(x):
    tensor_list.append(x)
    return tensor_list
append_tensor(tf.constant(1.2))
append_tensor(tf.constant(3.6))
tf.print(tensor_list)

@tf.function
def append_tensor_fun(x):
    tensor_list.append(x)
    return tensor_list
append_tensor_fun(tf.constant(5.661))
append_tensor_fun(tf.constant(8.552))
print(tensor_list)