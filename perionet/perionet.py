import tensorflow as tf
import numpy as np


def perioNet(
        inputs: tf.Tensor,
        order: int,
        periodicity_initializer: tf.initializers = tf.initializers.
        glorot_normal(),
        amplitude_initializer: tf.initializers = tf.initializers.glorot_normal(
        )):
    if inputs.shape.ndims != 2:
        raise Exception(
            "perioNet expects a tensor of 2 dimensions - {} was given".format(
                inputs.shape))

    batch, dims = inputs.shape.dims
    """ Not quite sure what the appropriate way of doing this is.. but for a hobby project this will do. """
    with tf.variable_scope(str(np.random.rand())):
        ts = tf.TensorShape([dims.value, order])
        cos_perio = tf.get_variable(
            "cos_periodicities", ts, initializer=periodicity_initializer)
        sin_perio = tf.get_variable(
            "sin_periodicities", ts, initializer=periodicity_initializer)
        cos_ampli = tf.get_variable(
            "cos_amplitudes", ts, initializer=amplitude_initializer)
        sin_ampli = tf.get_variable(
            "sin_amplitudes", ts, initializer=amplitude_initializer)

        ins = tf.split(inputs, dims.value, 1)

        cos_perio = tf.split(cos_perio, dims.value, 0)
        sin_perio = tf.split(sin_perio, dims.value, 0)
        cos_ampli = tf.split(cos_ampli, dims.value, 0)
        sin_ampli = tf.split(sin_ampli, dims.value, 0)

        outs = []
        for i, cos_p, sin_p, cos_a, sin_a in zip(ins, cos_perio, sin_perio,
                                                 cos_ampli, sin_ampli):

            outs.append(
                tf.expand_dims(
                    tf.reduce_sum(cos_a * tf.math.cos(i * cos_p * np.pi) +
                                  sin_a * tf.math.sin(i * sin_p * np.pi), 1),
                    1))

        outs = tf.concat(outs, 1)

    return outs