import tensorflow as tf
if __name__ == '__main__':
    gpus = tf.config.list_physical_devices(device_type="GPU")
    print("GPU:", gpus)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("GPU:", gpus)