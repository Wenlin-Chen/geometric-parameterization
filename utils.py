import tensorflow as tf
import math
import matplotlib.pyplot as plt


@tf.function
def hyperspherical_to_cartesian(angles, radius):
    '''
        angles: [n_in-1, n_out], each column is a directional vector represented by (n_in-1) angular parameters
        radius: [1, n_out] or scalar (broadcast), radius for each direction
        
        return n_out cartesian coordinates: [n_in, n_out]
    '''
    pad_shape = [1, angles.shape[-1]]
    augmented_angles = tf.concat([tf.zeros(pad_shape, dtype=angles.dtype), angles], axis=-2)
    
    cos_angles = tf.math.cos(augmented_angles)
    rearranged_cos_angles = tf.roll(cos_angles, shift=-1, axis=-2)
    
    sin_angles = tf.math.sin(augmented_angles)
    augmented_sin_angles = tf.tensor_scatter_nd_update(
        sin_angles, 
        tf.constant([[0]], dtype=tf.int32),
        tf.ones(pad_shape, dtype=sin_angles.dtype)
    )
    accumulated_sin_angles = tf.math.cumprod(augmented_sin_angles, axis=-2)
    
    outputs = radius * accumulated_sin_angles * rearranged_cos_angles
    return outputs


@tf.function
def cartesian_to_hyperspherical(x):
    '''
        x: [n_in, n_out], each column is a vector in cartesian coordinate
        
        return n_out sets of angles: [n_in-1, n_out] and n_out radius
    '''
    numerator = x[:-1, :]
    correction = tf.sqrt(tf.reduce_sum(x[-2:, :]**2, axis=0, keepdims=True))
    augmented_numerator = tf.tensor_scatter_nd_update(
        numerator, 
        tf.constant([[-1]], dtype=tf.int32),
        numerator[-1:, :] + correction
    )
    denominator = tf.math.sqrt(tf.math.cumsum(x[1:, :]**2, reverse=True, axis=-2))
    angles = math.pi/2.0 - tf.math.atan2(augmented_numerator, denominator)
    augmented_angles = tf.tensor_scatter_nd_update(
        angles, 
        tf.constant([[-1]], dtype=tf.int32),
        2.0*angles[-1:, :]
    )

    radius = tf.norm(x, axis=0, keepdims=True)

    return augmented_angles, radius


def plot_loss(history, ylim, loss_metric_name="Loss", plot_validation=True):
    plt.plot(history.history['loss'], label='train_loss')
    if plot_validation:
        plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim(ylim)
    plt.xlabel('Epoch')
    plt.ylabel(loss_metric_name)
    plt.legend()
    plt.grid(True)
