import sys
sys.path.append("../")

import os

import numpy as np
import pandas as pd
import random
import tensorflow as tf
import numpy as np

from models import DenseReparam, DenseWN
from sklearn.model_selection import train_test_split

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

####################
# argv 1: dataset name
# argv 2: seed
####################
dataset_name = sys.argv[1]
GLOBAL_SEED = int(sys.argv[2])
####################

os.environ['PYTHONHASHSEED']=str(GLOBAL_SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU
dtype = 'float64'
tf.keras.backend.set_floatx(dtype)

if dataset_name == "boston":
    from tensorflow.keras.datasets import boston_housing
    (train_features, train_labels), (test_features, test_labels) = boston_housing.load_data()
    train_labels = np.reshape(train_labels, [-1 ,1])
    test_labels = np.reshape(test_labels, [-1 ,1])
    features = np.concatenate([train_features, test_features], axis=0)
    labels = np.concatenate([train_labels, test_labels], axis=0)

elif dataset_name == "concrete":
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
    column_names = ['Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water', 'Superplasticizer',
                    'Coarse_Aggregate', 'Fine_Aggregate', 'Age', 'Concrete_compressive_strength']
    raw_dataset = pd.read_excel(url, names=column_names)
    dataset = raw_dataset.copy()
    features = dataset.dropna()
    labels = features.pop('Concrete_compressive_strength')
    features = np.array(features)
    labels = np.array(labels).reshape(-1, 1)

elif dataset_name == "energy":
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx'
    raw_dataset = pd.read_excel(url, header=0)
    dataset = raw_dataset.copy()
    features = dataset.dropna()
    labels = features.pop('Y1')
    _ = features.pop("Y2")
    features = np.array(features)
    labels = np.array(labels).reshape(-1, 1)

elif dataset_name == "naval":
    url = urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip")
    zipped_dataset = ZipFile(BytesIO(url.read()))
    dataset = zipped_dataset.open('UCI CBM Dataset/data.txt')
    col_names = ["{}".format(i) for i in range(18)]
    raw_dataset = pd.read_csv(dataset, header=None, sep='   ', names=col_names)
    dataset = raw_dataset.copy()
    features = dataset.dropna()
    labels = features.pop('16')
    _ = features.pop('17')
    features = np.array(features)
    labels = np.array(labels).reshape(-1, 1)

elif dataset_name == "power":
    url = urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip")
    zipped_dataset = ZipFile(BytesIO(url.read()))
    raw_dataset = pd.read_excel(zipped_dataset.open('CCPP/Folds5x2_pp.xlsx'), header=0)
    dataset = raw_dataset.copy()
    features = dataset.dropna()
    labels = features.pop('PE')
    features = np.array(features)
    labels = np.array(labels).reshape(-1, 1)

elif dataset_name == "wine":
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    raw_dataset = pd.read_csv(url, header=0, delimiter=';')
    dataset = raw_dataset.copy()
    features = dataset.dropna()
    labels = features.pop('quality')
    features = np.array(features)
    labels = np.array(labels).reshape(-1, 1)

elif dataset_name == "yacht":
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data'
    col_names = ["{}".format(i) for i in range(7)]
    raw_dataset = pd.read_csv(url, header=None, delimiter='\s+', names=col_names)
    dataset = raw_dataset.copy()
    features = dataset.dropna()
    labels = features.pop('6')
    features = np.array(features)
    labels = np.array(labels).reshape(-1, 1)

else:
    raise ValueError("UCI dataset name not recognized!")

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=GLOBAL_SEED)

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

units = 100
n_epochs = 5000
batch_size = 1024
early_stop_patience = 100

lr_gmp = 0.1
lr_others = 0.01

os.makedirs("./outputs", exist_ok=True)


# Geometric Parameterization
def build_reparam_dnn_model(lr):
    model = tf.keras.Sequential([
        normalizer,
        DenseReparam(units, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(lr), metrics=[tf.keras.metrics.MeanSquaredError()])
    return model

reparam_dnn_model = build_reparam_dnn_model(lr=lr_gmp)

lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mean_squared_error', factor=0.1, patience=early_stop_patience//2, verbose=0, mode='min')
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=early_stop_patience, verbose=0, mode='min', restore_best_weights=True)

history_reparam_dnn = reparam_dnn_model.fit(
    train_features,
    train_labels,
    batch_size=batch_size,
    validation_data=(test_features, test_labels),
    verbose=0, 
    epochs=n_epochs,
    callbacks=[lr_decay, early_stop]
)

test_results_reparam_dnn = reparam_dnn_model.evaluate(test_features, test_labels, verbose=0)
reparam_result = "GmP Test RMSE (seed {}): {}".format(GLOBAL_SEED, np.sqrt(test_results_reparam_dnn[1]))
print(reparam_result)
with open("outputs/uci_{}.txt".format(dataset_name), "a") as myfile:
    myfile.write(reparam_result+'\n')

# Standard Parameterization
def build_dnn_model(lr):
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(lr), metrics=[tf.keras.metrics.MeanSquaredError()])
    return model

dnn_model = build_dnn_model(lr=lr_others)

lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mean_squared_error', factor=0.1, patience=early_stop_patience//2, verbose=0, mode='min')
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=early_stop_patience, verbose=0, mode='min', restore_best_weights=True)

history_dnn = dnn_model.fit(
    train_features,
    train_labels,
    batch_size=batch_size,
    validation_data=(test_features, test_labels),
    verbose=0, 
    epochs=n_epochs,
    callbacks=[lr_decay, early_stop]
)

test_results_dnn = dnn_model.evaluate(test_features, test_labels, verbose=0)
dnn_result = "SP Test RMSE (seed {}): {}".format(GLOBAL_SEED, np.sqrt(test_results_dnn[1]))
print(dnn_result)
with open("outputs/uci_{}.txt".format(dataset_name), "a") as myfile:
    myfile.write(dnn_result+'\n')

# Batch Normalization
def build_bn_dnn_model(lr):
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(units),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(lr), metrics=[tf.keras.metrics.MeanSquaredError()])
    return model

bn_dnn_model = build_bn_dnn_model(lr=lr_others)

lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mean_squared_error', factor=0.1, patience=early_stop_patience//2, verbose=0, mode='min')
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=early_stop_patience, verbose=0, mode='min', restore_best_weights=True)

history_bn_dnn = bn_dnn_model.fit(
    train_features,
    train_labels,
    batch_size=batch_size,
    validation_data=(test_features, test_labels),
    verbose=0, 
    epochs=n_epochs,
    callbacks=[lr_decay, early_stop]
)

test_results_bn_dnn = bn_dnn_model.evaluate(test_features, test_labels, verbose=0)
bn_result = "BN Test RMSE (seed {}): {}".format(GLOBAL_SEED, np.sqrt(test_results_bn_dnn[1]))
print(bn_result)
with open("outputs/uci_{}.txt".format(dataset_name), "a") as myfile:
    myfile.write(bn_result+'\n')

# Weight Normalization
def build_wn_dnn_model(lr):
    model = tf.keras.Sequential([
        normalizer,
        DenseWN(units, activation='relu'),
        DenseWN(1)
    ])
    
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(lr), metrics=[tf.keras.metrics.MeanSquaredError()])
    return model

wn_dnn_model = build_wn_dnn_model(lr=lr_others)

lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mean_squared_error', factor=0.1, patience=early_stop_patience//2, verbose=0, mode='min')
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=early_stop_patience, verbose=0, mode='min', restore_best_weights=True)

history_wn_dnn = wn_dnn_model.fit(
    train_features,
    train_labels,
    batch_size=batch_size,
    validation_data=(test_features, test_labels),
    verbose=0, 
    epochs=n_epochs,
    callbacks=[lr_decay, early_stop]
)

test_results_wn_dnn = wn_dnn_model.evaluate(test_features, test_labels, verbose=0)
wn_result = "WN Test RMSE (seed {}): {}".format(GLOBAL_SEED, np.sqrt(test_results_wn_dnn[1]))
print(wn_result)
with open("outputs/uci_{}.txt".format(dataset_name), "a") as myfile:
    myfile.write(wn_result+'\n')


with open("outputs/uci_{}.txt".format(dataset_name), "a") as myfile:
    myfile.write('\n')
