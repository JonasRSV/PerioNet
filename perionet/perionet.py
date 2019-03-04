import tensorflow as tf
import numpy as np


class perioInit(tf.keras.initializers.Initializer):
    def __init__(self):
        pass

    def __call__(self, shape, dtype=None, partition_info=None):
        dim = None
        order = None
        if isinstance(shape, tf.TensorShape):
            if shape.ndims != 2:
                raise Exception(
                    "PerioInit Expects shape of dim 2 - {} was given".format(
                        shape.ndims))

            dim = shape.dims[0].value
            order = shape.dims[1].value
        else:
            if len(shape) != 2:
                raise Exception(
                    "PerioInit Expects shape of dim 2 - {} was given".format(
                        shape.ndims))

            dim = shape[0]
            order = shape[1]

        return tf.Variable(
            np.array([np.arange(order) for _ in range(dim)]), dtype=dtype)


def perioNet(inputs: tf.Tensor,
             order: int,
             smoothing: float,
             periodicity_initializer: tf.initializers = perioInit(),
             amplitude_initializer: tf.initializers = tf.initializers.
             glorot_uniform()):
    if inputs.shape.ndims != 2:
        raise Exception(
            "perioNet expects a tensor of 2 dimensions - {} was given".format(
                inputs.shape))

    batch, dims = inputs.shape.dims
    """ Not quite sure what the appropriate way of doing this is.. but for a hobby project this will do. """
    with tf.variable_scope(str(np.random.rand())):
        ts = tf.TensorShape([dims.value, order])
        perio = tf.get_variable(
            "cos_periodicities", ts,
            initializer=periodicity_initializer) / smoothing
        cos_ampli = tf.get_variable(
            "cos_amplitudes", ts, initializer=amplitude_initializer)
        sin_ampli = tf.get_variable(
            "sin_amplitudes", ts, initializer=amplitude_initializer)

        ins = tf.split(inputs, dims.value, 1)

        perio = tf.split(perio, dims.value, 0)
        cos_ampli = tf.split(cos_ampli, dims.value, 0)
        sin_ampli = tf.split(sin_ampli, dims.value, 0)

        outs = []
        for i, p, cos_a, sin_a in zip(ins, perio, cos_ampli, sin_ampli):

            outs.append(
                tf.expand_dims(
                    tf.reduce_sum(cos_a * tf.math.cos(i * p * np.pi) +
                                  sin_a * tf.math.sin(i * p * np.pi), 1), 1))

        outs = tf.concat(outs, 1)

    return outs
