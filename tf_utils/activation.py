import tensorflow as tf


def get_activation_function(function, args):
    return {
        'sigmoid': tf.sigmoid(args),
        'rect_linear': tf.nn.relu(args),
        'None': args
    }[function]
