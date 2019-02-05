import numpy as np
import os
import tensorflow as tf
from tensorflow.python.keras.layers import InputLayer, Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense, LSTM
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.saved_model.signature_constants import PREDICT_INPUTS

from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn
from tensorflow.python.estimator.export.export_output import PredictOutput


INPUT_TENSOR_NAME = "PREDICT_INPUTS_input" # needs to match the name of the first layer + "_input"


def keras_model_fn(hyperparameters):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(256, return_sequences=True,input_shape=(1,16), activation='relu', name='PREDICT_INPUTS'))
    model.add(tf.keras.layers.Dropout(0.9))
    model.add(tf.keras.layers.LSTM(256))
    model.add(tf.keras.layers.Dropout(0.9))
    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
    
    _model = tf.keras.Model(inputs=model.input, outputs=model.output)
    
    _model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy'])
    return _model

def serving_input_fn(params):
    tensor = tf.placeholder(tf.float32, shape=[1,1,16])
    return build_raw_serving_input_receiver_fn({INPUT_TENSOR_NAME: tensor})()


def train_input_fn(training_dir, params):
    return _input_fn(training_dir, 'Train_Clean.csv')


def eval_input_fn(training_dir, params):
    return _input_fn(training_dir, 'Test_Clean.csv')


def _input_fn(training_dir, training_filename):
    training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=os.path.join(training_dir, training_filename), target_dtype=np.float32, features_dtype=np.float32)

    return tf.estimator.inputs.numpy_input_fn(       
        x={INPUT_TENSOR_NAME: np.array(training_set.data).reshape((-1,1,16))},
        y=np.asarray(training_set.target).reshape((-1,1)),
        num_epochs=None,
        shuffle=True)()
