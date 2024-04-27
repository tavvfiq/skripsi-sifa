import numpy as np
import tensorflow as tf
import gpu

gpu.set_memory_limit(1024)

GIGA = tf.pow(1024, 3)
x = tf.ones(shape=(GIGA, tf.int8.size, 2), dtype=tf.int8)