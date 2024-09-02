import tensorflow as tf
import math
from utils import hyperspherical_to_cartesian, cartesian_to_hyperspherical


class GlorotUniformHyperSpherical(tf.keras.initializers.Initializer):
    # Transform the glorit uniform initialization in the weight space to spatial locations of ReLU features (n_in >= 2)
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, shape, dtype=None):
        n_in = shape[-2] - 1 
        n_out = shape[-1]
        assert n_in > 1

        threshold = math.sqrt(6.0 / (n_in + n_out))
        weight_init = tf.random.uniform([n_in, n_out], minval=-1.0*threshold, maxval=threshold, dtype=dtype, seed=self.seed)
        bias_init = tf.zeros([1, n_out], dtype=dtype)
        angles, radius = cartesian_to_hyperspherical(weight_init)
        lambda_init = bias_init / radius

        return tf.concat([angles, lambda_init, radius], axis=0)

class GlorotUniformHyperSpherical1D(tf.keras.initializers.Initializer):
    # Transform the glorit uniform initialization in the weight space to spatial locations of ReLU features (n_in == 1)
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, shape, dtype=None):
        n_in = shape[-2] - 2
        n_out = shape[-1]
        assert n_in == 1

        threshold = math.sqrt(6.0 / (n_in + n_out))
        weight_init = tf.random.uniform([n_in, n_out], minval=-1.0*threshold, maxval=threshold, dtype=dtype, seed=self.seed)
        #bias_init = tf.random.normal([1, n_out], mean=0.0, stddev=0.01, dtype=dtype, seed=self.seed)
        bias_init = 0.0
        radius = tf.math.abs(weight_init)
        lambda_init = bias_init / radius

        return tf.concat([weight_init, lambda_init, radius], axis=0)


class DenseReparam(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, initializer=GlorotUniformHyperSpherical(seed=None), initializer_1d=GlorotUniformHyperSpherical1D(seed=None), trainable_params=True, **kwargs):
        super(DenseReparam, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.initializer = initializer
        self.initializer_1d = initializer_1d
        self.trainable_params = trainable_params

    def build(self, input_shape):

        self.n_in = input_shape[-1]

        if self.n_in > 1:

            self.theta_lambda = self.add_weight(
                shape=(self.n_in+1, self.units),
                initializer=self.initializer,
                trainable=self.trainable_params,
                dtype=self.dtype,
                name="theta_lambda"
            )

        else:
            self.weight_lambda = self.add_weight(
                shape=(3, self.units), 
                initializer=self.initializer_1d, 
                trainable=self.trainable_params,
                dtype=self.dtype,
                name="weight_lambda"
            )
            

    def call(self, inputs):
        if self.n_in > 1:
            v = hyperspherical_to_cartesian(self.theta_lambda[:-2, :], radius=tf.cast(1.0, dtype=self.dtype))
            z = tf.matmul(inputs, v) + self.theta_lambda[-2:-1, :]
            r = self.theta_lambda[-1:, :]
        else:
            z = tf.stop_gradient(self.weight_lambda[:-2, :] / tf.math.abs(self.weight_lambda[:-2, :])) * inputs + self.weight_lambda[-2:-1, :]
            r = self.weight_lambda[-1:, :]

        if self.activation is None:
            return r * z
        elif self.activation == "relu":
            return r * tf.nn.relu(z)
        else:
            raise ValueError


class GlorotUniformHyperWN(tf.keras.initializers.Initializer):
    # Transform the glorit uniform initialization in the weight space to spatial locations of ReLU features (n_in >= 2)
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, shape, dtype=None):
        n_in = shape[-2] - 2
        n_out = shape[-1]

        threshold = math.sqrt(6.0 / (n_in + n_out))
        weight_init = tf.random.uniform([n_in, n_out], minval=-1.0*threshold, maxval=threshold, dtype=dtype, seed=self.seed)
        bias_init = tf.zeros([1, n_out], dtype=dtype)
        length = tf.norm(weight_init, axis=0, keepdims=True)

        return tf.concat([weight_init, bias_init, length], axis=0)


class DenseWN(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, initializer=GlorotUniformHyperWN(seed=None), trainable_params=True, **kwargs):
        super(DenseWN, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.initializer = initializer
        self.trainable_params = trainable_params

    def build(self, input_shape):

        self.n_in = input_shape[-1]

        self.weight_bias_length = self.add_weight(
            shape=(self.n_in+2, self.units),
            initializer=self.initializer,
            trainable=self.trainable_params,
            dtype=self.dtype,
            name="weight_bias_length"
        )

    def call(self, inputs):
        v = self.weight_bias_length[:-2, :] / tf.norm(self.weight_bias_length[:-2, :], axis=0, keepdims=True)
        z = self.weight_bias_length[-1:, :] * tf.matmul(inputs, v) + self.weight_bias_length[-2:-1, :]

        if self.activation is None:
            return z
        elif self.activation == "relu":
            return tf.nn.relu(z)
        else:
            raise ValueError


class Conv2DReparam(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='VALID', activation=None, **kwargs):
        super(Conv2DReparam, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation

    def build(self, input_shape):

        self.n_in = input_shape[-1]

        self.theta_lambda = self.add_weight(
            shape=(self.n_in*self.kernel_size[0]*self.kernel_size[1]+1, self.filters),
            initializer=GlorotUniformHyperSpherical(seed=None),
            trainable=True,
            dtype=self.dtype,
            name="theta_lambda"
        )

    def call(self, inputs):
        v = hyperspherical_to_cartesian(self.theta_lambda[:-2, :], radius=tf.cast(1.0, dtype=self.dtype))
        kernel = tf.reshape(v, [self.kernel_size[0], self.kernel_size[1], self.n_in, self.filters])
        z = tf.nn.conv2d(inputs, kernel, self.strides, self.padding) + tf.squeeze(self.theta_lambda[-2:-1, :])
        r = self.theta_lambda[-1:, :]

        if self.activation is None:
            return r * z
        elif self.activation == "relu":
            return r * tf.nn.relu(z)
        else:
            raise ValueError


class Conv2DWN(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='VALID', activation=None, **kwargs):
        super(Conv2DWN, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation

    def build(self, input_shape):

        self.n_in = input_shape[-1]

        self.weight_bias_length = self.add_weight(
            shape=(self.n_in*self.kernel_size[0]*self.kernel_size[1]+2, self.filters),
            initializer=GlorotUniformHyperWN(seed=None),
            trainable=True,
            dtype=self.dtype,
            name="theta_lambda"
        )

    def call(self, inputs):
        v = self.weight_bias_length[:-2, :] / tf.norm(self.weight_bias_length[:-2, :], axis=0, keepdims=True)
        kernel = tf.reshape(v, [self.kernel_size[0], self.kernel_size[1], self.n_in, self.filters])
        z = self.weight_bias_length[-1:, :] * tf.nn.conv2d(inputs, kernel, self.strides, self.padding) + tf.squeeze(self.weight_bias_length[-2:-1, :])

        if self.activation is None:
            return z
        elif self.activation == "relu":
            return tf.nn.relu(z)
        else:
            raise ValueError
