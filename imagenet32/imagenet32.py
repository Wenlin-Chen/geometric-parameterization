import os
import sys
sys.path.append("../")

import numpy as np
import random
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

from models import Conv2DReparam, Conv2DWN, DenseWN
from mean_norm import MeanOnlyBatchNormalization

####################
# argv 1: seed
# argv 2: batch size
# argv 3: parameterization name
####################
GLOBAL_SEED = int(sys.argv[1])
batch_size = int(sys.argv[2])
parameterization = sys.argv[3] # gmp, gmp_mn, standard, batchnorm, weightnorm, weightnorm_mbn
####################

os.environ['PYTHONHASHSEED']=str(GLOBAL_SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU
dtype = 'float32'
tf.keras.backend.set_floatx(dtype)

ds_train = tfds.load(
    'imagenet_resized/32x32',
    split="train",
    as_supervised=True,
)

ds_val = tfds.load(
    'imagenet_resized/32x32',
    split="validation",
    as_supervised=True,
)

train_mean = tf.constant([0.481098, 0.45747134, 0.40785483], dtype=dtype)

def preprocess_img(image, label):
    """Normalizes images: `uint8` -> `dtype`."""
    normalized = tf.cast(image, dtype) / 255.0
    standardized = normalized - train_mean
    return standardized, label


ds_train = ds_train.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_val = ds_val.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)

import wandb
wandb.init(project="ImageNet32", group=parameterization, name="seed{}-batchsize{}".format(GLOBAL_SEED, batch_size))

ds_train = ds_train.cache()
ds_train = ds_train.shuffle(100000)
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_val = ds_val.cache()
ds_val = ds_val.batch(batch_size)
ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

ds_test = ds_val


def build_model(parameterization):
    if parameterization == "gmp":
        model = tf.keras.Sequential([
            # data augmentation
            tf.keras.layers.RandomFlip(mode="horizontal"),
            # block 1
            Conv2DReparam(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu'),
            Conv2DReparam(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            # block 2
            Conv2DReparam(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu'),
            Conv2DReparam(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            # block 3
            Conv2DReparam(filters=512, kernel_size=(3, 3), padding='SAME', activation='relu'),
            Conv2DReparam(filters=512, kernel_size=(3, 3), padding='SAME', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            # top
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1000),
        ])
    elif parameterization == "gmp_mn":
        model = tf.keras.Sequential([
            # data augmentation
            tf.keras.layers.RandomFlip(mode="horizontal"),
            # block 1
            Conv2DReparam(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu'),
            MeanOnlyBatchNormalization(),
            Conv2DReparam(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu'),
            MeanOnlyBatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            # block 2
            Conv2DReparam(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu'),
            MeanOnlyBatchNormalization(),
            Conv2DReparam(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu'),
            MeanOnlyBatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            # block 3
            Conv2DReparam(filters=512, kernel_size=(3, 3), padding='SAME', activation='relu'),
            MeanOnlyBatchNormalization(),
            Conv2DReparam(filters=512, kernel_size=(3, 3), padding='SAME', activation='relu'),
            MeanOnlyBatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            # top
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1000),
        ])
    elif parameterization == 'standard':
        model = tf.keras.Sequential([
            # data augmentation
            tf.keras.layers.RandomFlip(mode="horizontal"),
            # block 1
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu'),
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            # block 2
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu'),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            # block 3
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='SAME', activation='relu'),
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='SAME', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            # top
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1000),
        ])

    elif parameterization == 'batchnorm':
        model = tf.keras.Sequential([
            # data augmentation
            tf.keras.layers.RandomFlip(mode="horizontal"),
            # block 1
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='SAME'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='SAME'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            # block 2
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='SAME'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='SAME'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            # block 3
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='SAME'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='SAME'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            # top
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1000),
        ])

    elif parameterization == 'weightnorm':
        model = tf.keras.Sequential([
            # data augmentation
            tf.keras.layers.RandomFlip(mode="horizontal"),
            # block 1
            Conv2DWN(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu'),
            Conv2DWN(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            # block 2
            Conv2DWN(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu'),
            Conv2DWN(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            # block 3
            Conv2DWN(filters=512, kernel_size=(3, 3), padding='SAME', activation='relu'),
            Conv2DWN(filters=512, kernel_size=(3, 3), padding='SAME', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            # top
            tf.keras.layers.GlobalAveragePooling2D(),
            DenseWN(1000),
        ])
    elif parameterization == 'weightnorm_mbn':
        model = tf.keras.Sequential([
            # data augmentation
            tf.keras.layers.RandomFlip(mode="horizontal"),
            # block 1
            Conv2DWN(filters=128, kernel_size=(3, 3), padding='SAME'),
            MeanOnlyBatchNormalization(),
            tf.keras.layers.ReLU(),
            Conv2DWN(filters=128, kernel_size=(3, 3), padding='SAME'),
            MeanOnlyBatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            # block 2
            Conv2DWN(filters=256, kernel_size=(3, 3), padding='SAME'),
            MeanOnlyBatchNormalization(),
            tf.keras.layers.ReLU(),
            Conv2DWN(filters=256, kernel_size=(3, 3), padding='SAME'),
            MeanOnlyBatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            # block 3
            Conv2DWN(filters=512, kernel_size=(3, 3), padding='SAME'),
            MeanOnlyBatchNormalization(),
            tf.keras.layers.ReLU(),
            Conv2DWN(filters=512, kernel_size=(3, 3), padding='SAME'),
            MeanOnlyBatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            # top
            tf.keras.layers.GlobalAveragePooling2D(),
            DenseWN(1000),
        ])

    else:
        raise ValueError("Unknown parameterization")

    init_lr = 0.1 if parameterization=="gmp" or parameterization=="gmp_mn" else 0.01

    model.compile(
        optimizer=tf.keras.optimizers.SGD(init_lr, momentum=0.9),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)],
    )
    return model


def wandb_log_func(epoch, logs):
    train_loss = logs['loss']
    train_acc = logs['sparse_categorical_accuracy']
    train_acc_topk = logs['sparse_top_k_categorical_accuracy']
    val_loss = logs['val_loss']
    val_acc = logs['val_sparse_categorical_accuracy']
    val_acc_topk = logs['val_sparse_top_k_categorical_accuracy']
    wandb.log(
        dict(
            train_loss=train_loss,
            train_acc=train_acc, 
            train_acc_top5=train_acc_topk, 
            val_loss=val_loss,
            val_acc=val_acc, 
            val_acc_top5=val_acc_topk, 
            learning_rate=model.optimizer.lr.numpy(),
        )
    )

model = build_model(parameterization)

wandb.log(
        dict(
            learning_rate=model.optimizer.lr.numpy(),
        )
    )

wandb_log_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=wandb_log_func)
lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_sparse_top_k_categorical_accuracy', factor=0.1, patience=5, verbose=1, mode='max')
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_top_k_categorical_accuracy', patience=10, verbose=0, mode='max', restore_best_weights=True)

history = model.fit(ds_train, batch_size=batch_size, epochs=500, validation_data=ds_val, verbose=1, callbacks=[wandb_log_callback, lr_decay, early_stop])
    