import tensorflow as tf


def get_optimizer(optimizer, learning_rate) :
    return {
        'gradient_descent': tf.train.GradientDescentOptimizer(learning_rate),
        'adam': tf.train.AdamOptimizer(learning_rate),
        'adagrad': tf.train.AdagradOptimizer(learning_rate)
    }[optimizer]