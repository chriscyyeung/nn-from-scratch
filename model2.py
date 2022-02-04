"""Implementation of a simple multilayer neural network using TensorFlow as
part of Assignment 1 of CISC 874 Neural and Cognitive Computing.

Author: Chris Yeung
Student number: 20055209
Date modified: 2/3/2022
"""

import os
import datetime
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from util import get_confusion_matrix, save_confusion_matrix

# Set random seed
tf.random.set_seed(2022)

NUM_CLASSES = 10


def main():
    # Load/preprocess the data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)

    # Leave 10000 samples for validation
    x_train, x_val = x_train[:50000], x_train[50000:]
    y_train, y_val = y_train[:50000], y_train[50000:]

    # One-hot encode class labels
    y_train = tf.one_hot(y_train, NUM_CLASSES)
    y_val = tf.one_hot(y_val, NUM_CLASSES)
    y_test = tf.one_hot(y_test, NUM_CLASSES)

    # Calculate number of hidden neurons
    n_hidden = (x_train.shape[1] + NUM_CLASSES) // 2

    # Build model
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(x_train.shape[1])),
            tf.keras.layers.Dense(
                n_hidden, activation="sigmoid", kernel_initializer="glorot_uniform"
            ),
            tf.keras.layers.Dense(10, activation="sigmoid", kernel_initializer="glorot_uniform")
        ]
    )

    # Training configuration
    model.compile(
        optimizer=tf.keras.optimizers.SGD(momentum=0.9),  # default lr is 0.01
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )

    # Callback for early stopping
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=3)]

    # Train model
    model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=1000,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=2
    )

    # Evaluate model on test data
    results = model.evaluate(x_test, y_test, batch_size=x_test.shape[0])
    print("test loss, test acc, test precision, test recall:", results)

    # Get confusion matrices
    save_path = "tf_model_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.mkdir(save_path)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    val_cm = get_confusion_matrix(val_pred, y_val.numpy())
    test_cm = get_confusion_matrix(test_pred, y_test.numpy())
    save_confusion_matrix(val_cm, save_path, NUM_CLASSES, "train")
    save_confusion_matrix(test_cm, save_path, NUM_CLASSES, "test")


if __name__ == '__main__':
    main()
