"""
优化器，TensorFlow中有三种使用方法
1、使用apply_gradient()，这种和pytorch比较统一，我也比较喜欢使用这种
2、直接使用optimizers.minimize()
3、使用model封装后，然后使用model.fit()方法
"""
"""
演示第一种方法apply_gradient
"""
import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# 求f(x) = a*x**2 + b*x + c的最小值

import tensorflow as tf

x = tf.Variable(0.0,name='x',dtype=tf.float32)
lr = 0.0001
optimizers = tf.keras.optimizers.SGD(learning_rate=lr)

@tf.function
def minimize_function():
    a = tf.constant(2.0)
    b = tf.constant(-5.0)
    c = tf.constant(99.0)

    while tf.constant(True):
        with tf.GradientTape() as tape:
            y = a * tf.pow(x,2) +b*x + c
        dy_dx = tape.gradient(y,x)
        optimizers.apply_gradients(grads_and_vars=[(dy_dx,x)])

        if tf.abs(dy_dx)<tf.constant(0.000001):
            break

        if tf.math.mod(optimizers.iterations,100)==0:
            tf.print('step=',optimizers.iterations)
            tf.print('dy_dx',dy_dx)
            tf.print('x=',x)
            tf.print('y=',y)
            tf.print('*'*100)
    y = a*tf.pow(x,2) + b*x + c
    return y
if __name__ == '__main__':
    tf.print('y=',minimize_function())
    tf.print('x=',x)