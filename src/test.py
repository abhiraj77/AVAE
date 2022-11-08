import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('XLA_GPU')))
physical_devices = tf.config.experimental.list_physical_devices('XLA_GPU')
print(physical_devices)
if physical_devices:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)