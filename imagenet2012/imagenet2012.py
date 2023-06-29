############################
# Go to `tensorflow/python/ops/image_ops_impl.py`` and modify the `_resize_images_common()` function as follows:
# scale_factor = `math_ops.minimum(scale_factor_height, scale_factor_width)`  --->  `scale_factor = math_ops.maximum(scale_factor_height, scale_factor_width)`
############################

import os
from datetime import datetime

import sys

import numpy as np
import random

import tensorflow as tf

import numpy as np
import tensorflow_datasets as tfds

from resnet34_batchnorm import build_resnet34_batchnorm
from resnet34_gmp import build_resnet34_gmp
from resnet34_weightnorm import build_resnet34_weightnorm
from utils import MyEarlyStopping, MyReduceLROnPlateau, center_crop

import wandb

####################
# arguments
####################
GLOBAL_SEED = int(sys.argv[1])
batch_size = int(sys.argv[2])
param_mode = str(sys.argv[3])  # gmp_mn, batchnorm, weightnorm_mbn
init_lr = float(sys.argv[4])
is_resume = bool(int(sys.argv[5]))
resume_id = str(sys.argv[6])
resume_time = str(sys.argv[7])
last_best = float(sys.argv[8])
es_wait = int(sys.argv[9])
ld_wait = int(sys.argv[10])

run_name = "seed_{}-batchsize_{}-{}".format(GLOBAL_SEED, batch_size, param_mode)
if is_resume:
    now_time = resume_time
else:
    now_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs("/rds/user/wc337/hpc-work/reparam/checkpoints/{}-{}".format(now_time, run_name))

config = {
    "seed": GLOBAL_SEED,
    "batch_size": batch_size,
    "init_lr": init_lr,
    "param_mode": param_mode,
}

if is_resume:
    wandb.init(
        project="ImageNet2012",
        entity='wc337', 
        group='resnet34_'+param_mode, 
        name=run_name, 
        config=config,
        id=resume_id,
        resume="must"
    )
else:
    wandb.init(
        project="ImageNet2012",
        entity='wc337', 
        group='resnet34_'+param_mode, 
        name=run_name, 
        config=config
    )


os.environ['PYTHONHASHSEED']=str(GLOBAL_SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)

dtype = 'float32'
tf.keras.backend.set_floatx(dtype)


strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

ds_train = tfds.load(
    'imagenet2012',
    data_dir="/rds/user/wc337/hpc-work/data/",
    split="train",
    as_supervised=True,
)

ds_val = tfds.load(
    'imagenet2012',
    data_dir="/rds/user/wc337/hpc-work/data/",
    split="validation",
    as_supervised=True,
)

# The data augmentation scheme here is a bit different from what we did for the paper, which should be better but computationally more expensive.
# 
# In the paper, we resize each origin image with aspect ratio preserved such that min(hight, width)=480 then central crop it to 480 x 480,
# which ensures that all images have the same size and thus they can be batched. During training, we random resize it to LxL with L~U[256, 480]
# then crop it to 224x224.
# 
# Here, we resize each origin image with aspect ratio preserved such that min(hight, width)=480 but do NOT central crop it. Note that each image is
# very likely to have a different size. During training, we random resize it to LxL with L~U[256, 480] then crop it to 224x224.

def preprocess_img(image, label):
    """Normalizes images: `uint8` -> `dtype`."""

    resize_size = tf.random.uniform(shape=[], minval=256, maxval=480, dtype=tf.int32)

    image = tf.cast(image, dtype=dtype)
    image = tf.image.resize(image, tf.stack([resize_size, resize_size]), preserve_aspect_ratio=True)
    image = tf.image.random_crop(image, [224, 224, 3])
    return image, label

def return_central_crop(original_size, target_size):
    edge = (original_size - target_size) // 2
    if (target_size - original_size) % 2 == 0:
        return edge, -edge
    else:
        return edge+1, -edge

def processes_val_img(image, label):
    image = tf.cast(image, dtype=dtype)
    image = tf.image.resize(image, [368, 368], preserve_aspect_ratio=True)
    image = center_crop(image, 224, 224)
    return image, label


ds_train = ds_train.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_val = ds_val.map(processes_val_img, num_parallel_calls=tf.data.AUTOTUNE)

#ds_train = ds_train.cache()
ds_train = ds_train.shuffle(10000)
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_val = ds_val.cache()
ds_val = ds_val.batch(batch_size)
ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

ds_test = ds_val

with strategy.scope():
    optimizer=tf.keras.optimizers.SGD(init_lr, momentum=0.9)
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)]
    if param_mode == "gmp_mn":
        model = build_resnet34_gmp(optimizer, loss, metrics)
    elif param_mode == "batchnorm":
        model = build_resnet34_batchnorm(optimizer, loss, metrics)
    elif param_mode == "weightnorm_mbn":
        model = build_resnet34_weightnorm(optimizer, loss, metrics)
    else:
        raise ValueError("Unknown parameterization")

if is_resume:
    with strategy.scope():
        model.load_weights("/rds/user/wc337/hpc-work/reparam/checkpoints/{}-{}/".format(now_time, run_name))
else:
    wandb.log(
        dict(
            learning_rate=model.optimizer.lr.numpy(),
        )
    )

print("Initial learning rate:", model.optimizer.lr.numpy())

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
        
wandb_log_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=wandb_log_func)
save_weights = tf.keras.callbacks.ModelCheckpoint(filepath="/rds/user/wc337/hpc-work/reparam/checkpoints/{}-{}/".format(now_time, run_name), save_freq="epoch", save_weights_only=True)

if is_resume:
    early_stop = MyEarlyStopping(monitor='val_sparse_top_k_categorical_accuracy', patience=10, verbose=0, mode='max', restore_best_weights=False, best=last_best, wait=es_wait)
    lr_decay = MyReduceLROnPlateau(monitor='val_sparse_top_k_categorical_accuracy', factor=0.1, patience=5, verbose=1, mode='max', best=last_best, wait=ld_wait)
else:
    early_stop = MyEarlyStopping(monitor='val_sparse_top_k_categorical_accuracy', patience=10, verbose=0, mode='max', restore_best_weights=False)
    lr_decay = MyReduceLROnPlateau(monitor='val_sparse_top_k_categorical_accuracy', factor=0.1, patience=5, verbose=1, mode='max')

history_model = model.fit(ds_train, batch_size=batch_size, epochs=10000, validation_data=ds_val, verbose=1, callbacks=[wandb_log_callback, lr_decay, save_weights, early_stop])
