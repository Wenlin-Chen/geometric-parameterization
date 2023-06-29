import sys
sys.path.append("../")

import tensorflow as tf
import numpy as np
from keras.layers import *
from keras import Model
from models import DenseWN, Conv2DWN
from keras import layers as Layers
from mean_norm import MeanOnlySyncBatchNormalization

val_target_size = tf.cast(368, dtype=tf.int32)
std = [0.229, 0.224, 0.225]
std_tensor = tf.constant(np.array(std), dtype=tf.float32)
eigval = tf.constant(np.array([55.46, 4.794, 1.148]), dtype=tf.float32)
eigvec = tf.constant(np.array([[-0.5836, -0.6948, 0.4203], [-0.5808, -0.0045, -0.8140], [-0.5675, 0.7192, 0.4009]]), dtype=tf.float32)


class ResBlockWeightNorm(Model):
    def __init__(self, channels, stride=1):
        super(ResBlockWeightNorm, self).__init__(name='ResBlockWeightNorm')
        self.flag = (stride != 1)
        self.conv1 = Conv2DWN(channels, [3, 3], stride, padding='SAME')
        self.bn1 = MeanOnlySyncBatchNormalization()
        self.conv2 = Conv2DWN(channels, [3, 3], padding='SAME')
        self.bn2 = MeanOnlySyncBatchNormalization()
        self.relu = ReLU()
        if self.flag:
            self.bn3 = MeanOnlySyncBatchNormalization()
            self.conv3 = Conv2DWN(channels, [1, 1], stride)

    def call(self, x, training=None):
        x1 = self.conv1(x)
        x1 = self.bn1(x1, training=training)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1, training=training)
        if self.flag:
            x = self.conv3(x)
            x = self.bn3(x, training=training)
        x1 = Layers.add([x, x1])
        x1 = self.relu(x1)
        return x1


class ResNet34WeightNorm(Model):
    def __init__(self):
        super(ResNet34WeightNorm, self).__init__(name='ResNet34WeightNorm')
        self.random_flip = tf.keras.layers.RandomFlip(mode="horizontal")

        self.conv1 = Conv2DWN(64, [7, 7], 2, padding='SAME')
        self.bn = MeanOnlySyncBatchNormalization()
        self.relu = ReLU()
        self.mp1 = MaxPooling2D(3, 2)

        self.conv2_1 = ResBlockWeightNorm(64)
        self.conv2_2 = ResBlockWeightNorm(64)
        self.conv2_3 = ResBlockWeightNorm(64)

        self.conv3_1 = ResBlockWeightNorm(128, 2)
        self.conv3_2 = ResBlockWeightNorm(128)
        self.conv3_3 = ResBlockWeightNorm(128)
        self.conv3_4 = ResBlockWeightNorm(128)

        self.conv4_1 = ResBlockWeightNorm(256, 2)
        self.conv4_2 = ResBlockWeightNorm(256)
        self.conv4_3 = ResBlockWeightNorm(256)
        self.conv4_4 = ResBlockWeightNorm(256)
        self.conv4_5 = ResBlockWeightNorm(256)
        self.conv4_6 = ResBlockWeightNorm(256)

        self.conv5_1 = ResBlockWeightNorm(512, 2)
        self.conv5_2 = ResBlockWeightNorm(512)
        self.conv5_3 = ResBlockWeightNorm(512)

        self.pool = GlobalAveragePooling2D()
        self.fc1 = DenseWN(1000)

    def call(self, x, training=None):
        if training:
            x = self.random_flip(x, training=training)
            alpha = tf.random.normal([3], mean=0.0, stddev=0.1)
            offset = tf.experimental.numpy.dot(eigvec * alpha, eigval)
            x = tf.clip_by_value(x + offset, 0.0, 255.0)

        x = tf.keras.applications.imagenet_utils.preprocess_input(x, mode='torch')
        x = x * std_tensor

        x = self.conv1(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        x = self.mp1(x)

        x = self.conv2_1(x, training=training)
        x = self.conv2_2(x, training=training)
        x = self.conv2_3(x, training=training)

        x = self.conv3_1(x, training=training)
        x = self.conv3_2(x, training=training)
        x = self.conv3_3(x, training=training)
        x = self.conv3_4(x, training=training)

        x = self.conv4_1(x, training=training)
        x = self.conv4_2(x, training=training)
        x = self.conv4_3(x, training=training)
        x = self.conv4_4(x, training=training)
        x = self.conv4_5(x, training=training)
        x = self.conv4_6(x, training=training)

        x = self.conv5_1(x, training=training)
        x = self.conv5_2(x, training=training)
        x = self.conv5_3(x, training=training)

        x = self.pool(x)
        x = self.fc1(x)
        return x


def build_resnet34_weightnorm(optimizer, loss, metrics):
    model = ResNet34WeightNorm()
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )
    return model
