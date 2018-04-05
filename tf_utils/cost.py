import tensorflow as tf

def get_cost(cost, args):
    return {
        'square': tf.square(args),
    }[cost]