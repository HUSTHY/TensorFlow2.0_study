"""
神经网络通常依赖反向传播求梯度来更新网络参数
TensorFlow使用梯度磁带tf.GradientTape来记录正向运算过程，然后使用反向传播来得到梯度值
这种机制叫做自动微分机制

"""



"""
利用梯度磁带求导数
"""
# f(x) = a*x**2 + b*x + c的导数,分别求a b c x 的微分

import tensorflow as tf
#
# x = tf.Variable(2.0,dtype=tf.float32,name='x')
# a = tf.constant(1.0)
# b = tf.constant(2.0)
# c = tf.constant(1.0)
#
# with tf.GradientTape() as tape:
#     tape.watch([a,b,c])
#     y = a*tf.pow(x,2)+b*x+c
#
# dy_dx, dy_da, dy_db, dy_dc = tape.gradient(y,[x,a,b,c])
# print(dy_dx, dy_da, dy_db, dy_dc)
#
#
# @tf.function
# def get_gradient(x):
#     a = tf.constant(1.0)
#     b = tf.constant(2.0)
#     c = tf.constant(1.0)
#
#     with tf.GradientTape() as tape:
#         tape.watch([a, b, c])
#         y = a * tf.pow(x, 2) + b * x + c
#     dy_dx = tape.gradient(y,x)
#     return (y,dy_dx)
#
# result = get_gradient(tf.Variable(0.0,dtype=tf.float32))
# print('result',result)



"""
利用梯度磁带和优化器求最小值
1、形式一使用tf.GradientTape()和optimizer.apply_gradients
"""
# 在autograph中完成最小值求解

x = tf.Variable(0.0,name = "x",dtype = tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

@tf.function
def get_gradient_apply_gradients():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)

    for _ in tf.range(10000):
        with tf.GradientTape() as tape:
            y =  a * tf.pow(x, 2) + b * x + c
        dy_dx = tape.gradient(y,x)
        optimizer.apply_gradients(grads_and_vars=[(dy_dx,x)])

    y = a * tf.pow(x, 2) + b * x + c
    return y

print(get_gradient_apply_gradients())
print(x)