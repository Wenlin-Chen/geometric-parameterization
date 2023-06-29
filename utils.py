import matplotlib.pyplot as plt
import tensorflow as tf
import math
import numpy as np
from tensorflow.python.platform import tf_logging as logging
from keras import backend
from keras.engine import base_layer
from keras.engine import base_preprocessing_layer
from keras.utils import control_flow_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.image_ops_impl import _is_tensor, crop_to_bounding_box,  _assert, _ImageDimensions, _CheckAtLeast3DImage

H_AXIS = -3
W_AXIS = -2

ResizeMethod = tf.image.ResizeMethod

_RESIZE_METHODS = {
    'bilinear': ResizeMethod.BILINEAR,
    'nearest': ResizeMethod.NEAREST_NEIGHBOR,
    'bicubic': ResizeMethod.BICUBIC,
    'area': ResizeMethod.AREA,
    'lanczos3': ResizeMethod.LANCZOS3,
    'lanczos5': ResizeMethod.LANCZOS5,
    'gaussian': ResizeMethod.GAUSSIAN,
    'mitchellcubic': ResizeMethod.MITCHELLCUBIC
}


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


class MyEarlyStopping(tf.keras.callbacks.EarlyStopping):

    def __init__(self, monitor="val_loss", min_delta=0, patience=0, verbose=0, mode="auto", baseline=None, restore_best_weights=False, best=None, wait=None):
        if best is not None and wait is not None:
            self.is_resume = True
            self.last_best = best
            self.last_wait = wait
        else:
            self.is_resume = False

        super().__init__(monitor, min_delta, patience, verbose, mode, baseline, restore_best_weights)

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        if self.is_resume:
            self.wait = self.last_wait
            self.best = self.last_best
        else:
            self.wait = 0
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.stopped_epoch = 0
        self.best_weights = None


class MyReduceLROnPlateau(tf.keras.callbacks.ReduceLROnPlateau):

    def __init__(self, monitor="val_loss", factor=0.1, patience=10, verbose=0, mode="auto", min_delta=0.0001, cooldown=0, min_lr=0, best=None, wait=None, **kwargs):
        if best is not None and wait is not None:
            self.is_resume = True
            self.last_best = best
            self.last_wait = wait
        else:
            self.is_resume = False
        
        super().__init__(monitor, factor, patience, verbose, mode, min_delta, cooldown, min_lr, **kwargs)

    def _reset(self):
        """Resets wait counter and cooldown counter."""
        if self.mode not in ['auto', 'min', 'max']:
            logging.warning('Learning rate reduction mode %s is unknown, '
                            'fallback to auto mode.', self.mode)
            self.mode = 'auto'
        if (self.mode == 'min' or (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

        if self.is_resume:
            self.wait = self.last_wait
            self.best = self.last_best

        

class RandomResize(base_layer.Layer):
    """Randomly vary the height of a batch of images during training.

    Adjusts the height of a batch of images by a random factor. The input
    should be a 3D (unbatched) or 4D (batched) tensor in the `"channels_last"`
    image data format.

    By default, this layer is inactive during inference.

    Args:
    factor: A positive float (fraction of original height), or a tuple of size 2
        representing lower and upper bound for resizing vertically. When
        represented as a single float, this value is used for both the upper and
        lower bound. For instance, `factor=(0.2, 0.3)` results in an output with
        height changed by a random amount in the range `[20%, 30%]`.
        `factor=(-0.2, 0.3)` results in an output with height changed by a random
        amount in the range `[-20%, +30%]. `factor=0.2` results in an output with
        height changed by a random amount in the range `[-20%, +20%]`.
    interpolation: String, the interpolation method. Defaults to `"bilinear"`.
        Supports `"bilinear"`, `"nearest"`, `"bicubic"`, `"area"`,
        `"lanczos3"`, `"lanczos5"`, `"gaussian"`, `"mitchellcubic"`.
    seed: Integer. Used to create a random seed.

    Input shape:
    3D (unbatched) or 4D (batched) tensor with shape:
    `(..., height, width, channels)`, in `"channels_last"` format.

    Output shape:
    3D (unbatched) or 4D (batched) tensor with shape:
    `(..., random_height, width, channels)`.
    """

    def __init__(self,
                factor,
                interpolation='bilinear',
                seed=None,
                **kwargs):
        self.factor = factor
        if isinstance(factor, (tuple, list)):
            self.lower = factor[0]
            self.upper = factor[1]
        else:
            self.lower = -factor
            self.upper = factor

        if self.upper < self.lower:
            raise ValueError('`factor` cannot have upper bound less than '
                            'lower bound, got {}'.format(factor))
        if self.lower < -1. or self.upper < -1.:
            raise ValueError('`factor` must have values larger than -1, '
                            'got {}'.format(factor))
        self.interpolation = interpolation
        self._interpolation_method = get_interpolation(interpolation)
        self.seed = seed
        self._rng = make_generator(self.seed)
        super(RandomResize, self).__init__(**kwargs)
        base_preprocessing_layer.keras_kpl_gauge.get_cell('RandomResize').set(True)

    def call(self, inputs, training=True):
        if training is None:
            training = backend.learning_phase()

        def random_height_inputs():
            """Inputs height-adjusted with random ops."""
            inputs_shape = tf.shape(inputs)
            img_hd = tf.cast(inputs_shape[H_AXIS], tf.float32)
            img_wd = tf.cast(inputs_shape[W_AXIS], tf.float32)
            factor = self._rng.uniform(
                shape=[],
                minval=(1.0 + self.lower),
                maxval=(1.0 + self.upper))
            adjusted_height = tf.cast(factor * img_hd, tf.int32)
            adjusted_width = tf.cast(factor * img_wd, tf.int32)
            adjusted_size = tf.stack([adjusted_height, adjusted_width])
            output = tf.image.resize(
                images=inputs, size=adjusted_size, method=self._interpolation_method)
            output_shape = inputs.shape.as_list()
            output_shape[H_AXIS] = None
            output.set_shape(output_shape)
            return output

        return control_flow_util.smart_cond(training, random_height_inputs,
                                        lambda: inputs)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        input_shape[H_AXIS] = None
        return tf.TensorShape(input_shape)

    def get_config(self):
        config = {
            'factor': self.factor,
            'interpolation': self.interpolation,
            'seed': self.seed,
        }
        base_config = super(RandomResize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

def make_generator(seed=None):
  """Creates a random generator.

  Args:
    seed: the seed to initialize the generator. If None, the generator will be
      initialized non-deterministically.

  Returns:
    A generator object.
  """
  if seed is not None:
    return tf.random.Generator.from_seed(seed)
  else:
    return tf.random.Generator.from_non_deterministic_state()


def get_interpolation(interpolation):
  interpolation = interpolation.lower()
  if interpolation not in _RESIZE_METHODS:
    raise NotImplementedError(
        'Value not recognized for `interpolation`: {}. Supported values '
        'are: {}'.format(interpolation, _RESIZE_METHODS.keys()))
  return _RESIZE_METHODS[interpolation]


def equal_(x, y):
    if _is_tensor(x) or _is_tensor(y):
        return math_ops.equal(x, y)
    else:
        return x == y

def center_crop(image, target_height, target_width):
 
    with ops.name_scope(None, 'center_crop', [image]):
        image = ops.convert_to_tensor(image, name='image')
        image_shape = image.get_shape()
        is_batch = True
        if image_shape.ndims == 3:
            is_batch = False
            image = array_ops.expand_dims(image, 0)
        elif image_shape.ndims is None:
            is_batch = False
            image = array_ops.expand_dims(image, 0)
            image.set_shape([None] * 4)
        elif image_shape.ndims != 4:
            raise ValueError(
                '\'image\' (shape %s) must have either 3 or 4 dimensions.' % image_shape)
        
        _, height, width, _ = _ImageDimensions(image, rank=4)

        assert_ops = _CheckAtLeast3DImage(image, require_static=False)
        assert_ops += _assert(target_width > 0, ValueError, 'target_width must be > 0.')
        assert_ops += _assert(target_width <= width, ValueError, 'target_width must be < width.')
        assert_ops += _assert(target_height > 0, ValueError, 'target_height must be > 0.')
        assert_ops += _assert(target_height <= height, ValueError, 'target_height must be < height.')

        image = control_flow_ops.with_dependencies(assert_ops, image)
        # `crop_to_bounding_box` and `pad_to_bounding_box` have their own checks.
        # Make sure our checks come first, so that error messages are clearer.
        if _is_tensor(target_height):
            target_height = control_flow_ops.with_dependencies(assert_ops, target_height)
        if _is_tensor(target_width):
            target_width = control_flow_ops.with_dependencies(assert_ops, target_width)

        width_diff = target_width - width
        offset_crop_width = -width_diff // 2

        height_diff = target_height - height
        offset_crop_height = -height_diff // 2

        # Center crop.
        cropped = crop_to_bounding_box(image, offset_crop_height, offset_crop_width, target_height, target_width)

        # In theory all the checks below are redundant.
        if cropped.get_shape().ndims is None:
            raise ValueError('cropped contains no shape.')

    _, cropped_height, cropped_width, _ = _ImageDimensions(cropped, rank=4)

    assert_ops = []
    assert_ops += _assert(
        equal_(cropped_height, target_height), ValueError,
        'cropped height is not correct.')
    assert_ops += _assert(
        equal_(cropped_width, target_width), ValueError,
        'cropped width is not correct.')

    cropped = control_flow_ops.with_dependencies(assert_ops, cropped)

    if not is_batch:
      cropped = array_ops.squeeze(cropped, axis=[0])

    return cropped


def plot_loss(history, ylim, loss_metric_name="Loss", plot_validation=True):
    plt.plot(history.history['loss'], label='train_loss')
    if plot_validation:
        plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim(ylim)
    plt.xlabel('Epoch')
    plt.ylabel(loss_metric_name)
    plt.legend()
    plt.grid(True)
