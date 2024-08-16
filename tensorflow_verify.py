import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# Create a simple tensor operation to check GPU availability
a = tf.constant([1.0, 2.0, 3.0])
b = tf.constant([1.0, 2.0, 3.0])
c = a + b
print("Result:", c)

# Verify GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
print("GPUs:", physical_devices)

if not physical_devices:
    print("No GPUs found. Ensure you have the TensorFlow-Metal plugin installed and properly configured.")
else:
    print("TensorFlow is using the Metal backend.")
