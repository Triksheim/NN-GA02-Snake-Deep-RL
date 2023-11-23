import tensorflow as tf

print("Is eager execution enabled: ", tf.executing_eagerly())

tf.compat.v1.enable_eager_execution()
print("Is eager execution enabled: ", tf.executing_eagerly())