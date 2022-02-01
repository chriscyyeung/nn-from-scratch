"""Implementation of a simple multilayer neural network as part of Assignment 1
of CISC 874 Neural and Cognitive Computing.

Author: Chris Yeung
Student number: 20055209
Date modified: 1/31/2022
"""

import os
import datetime
import tqdm
import math
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.datasets import mnist
np.seterr(divide="ignore", invalid="ignore")  # nans are handled

# Set random seed
rng = np.random.default_rng(2022)


class TwoLayerNN:
    """Class representing an artificial neural network with one hidden layer,
    using sigmoid activation and mean squared error loss. Weights are updated
    using gradient descent with momentum.
    """

    def __init__(self, input_size, n_hidden, n_output, learning_rate=0.01, momentum=0.9):
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.learning_rate = learning_rate
        self.momentum = momentum

        # Initialize weights and bias
        self.weights_hi = self.get_initial_weights(input_size, n_hidden)
        self.weights_jh = self.get_initial_weights(n_hidden, n_output)
        self.bias_i = np.zeros(n_hidden)
        self.bias_h = np.zeros(n_output)

        # Initialize arrays to store previous weight increments
        self.previous_delta_hi = np.zeros((input_size, n_hidden))
        self.previous_delta_jh = np.zeros((n_hidden, n_output))

        # Initialize layers
        self.input_layer = np.zeros(input_size)
        self.hidden_layer = np.zeros(n_hidden)
        self.output_layer = np.zeros(n_output)

        # Initialize variables to save best model weights
        self.best_weights_hi = None
        self.best_weights_jh = None
        self.best_bias_i = None
        self.best_bias_h = None

    @staticmethod
    def get_initial_weights(n_input, n_output):
        """Xavier weight initialization. Initializes weights by sampling from
        a uniform distribution within [-limit, limit], where limit =
        sqrt(6 / (number of input nodes + number of output nodes)).

        :return: an array of shape (n_input, n_output) of the weights
        """
        scale = 1 / max(1., (n_input + n_output) / 2.)
        limit = math.sqrt(3 * scale)
        return rng.uniform(-limit, limit, size=(n_input, n_output))

    @staticmethod
    def sigmoid(a):
        """Returns a new array after applying a sigmoid activation function.

        :param a: an N-D array of floating-point values
        :return: an array of the same shape as a of the activated values
        """
        # Use piecewise to prevent instability (inf)
        return np.piecewise(a, [a > 0], [lambda x: 1 / (1 + np.exp(-x)), lambda x: np.exp(x) / (1 + np.exp(x))])

    def sigmoid_derivative(self, a):
        """Returns an array of the values of a after applying the derivative
        of the sigmoid activation function.

        :param a: an N-D array of floating-point values
        :return: an array of same shape as a with sigmoid derivative applied
        """
        return self.sigmoid(a) * (1 - self.sigmoid(a))

    def set_best_weights(self):
        """Assigns the current model weights/biases as the best weights.

        :return: None
        """
        self.best_weights_hi = self.weights_hi
        self.best_weights_jh = self.weights_jh
        self.best_bias_i = self.bias_i
        self.best_bias_h = self.bias_h

    def forward(self, x):
        """Propagates the input batch through the network to generate
        a class prediction using a sigmoid activation.

        :param x: an array of shape (n_samples, n_input_nodes)
        :return: an array of shape (n_samples, n_classes) representing the
                 probability of the sample of belonging in a particular class
        """
        self.input_layer = x
        self.hidden_layer = self.sigmoid(np.dot(self.input_layer, self.weights_hi) + self.bias_i)
        self.output_layer = self.sigmoid(np.dot(self.hidden_layer, self.weights_jh) + self.bias_h)
        return self.output_layer

    def backprop(self, y):
        """Updates the weights and biases of the network using backpropagation.

        :param y: an array of shape (n_samples, n_classes) of the ground truth
        :return: None
        """
        # Calculate deltas
        delta_j = (y - self.output_layer) * self.sigmoid_derivative(self.output_layer)
        delta_h = delta_j @ self.weights_jh.T * self.sigmoid_derivative(self.hidden_layer)

        # Update weights with momentum
        d_weights_jh = self.learning_rate * (self.hidden_layer.T @ delta_j) + \
                       (self.momentum * self.previous_delta_jh)
        d_weights_hi = self.learning_rate * (self.input_layer.T @ delta_h) + \
                       (self.momentum * self.previous_delta_hi)
        self.weights_jh += d_weights_jh
        self.weights_hi += d_weights_hi

        # Update bias
        self.bias_h += self.learning_rate * np.sum(delta_j, axis=0)
        self.bias_i += self.learning_rate * np.sum(delta_h, axis=0)

        # Store weight increments
        self.previous_delta_jh = d_weights_jh
        self.previous_delta_hi = d_weights_hi


class Model:
    """Class for training and testing the two-layer neural network."""

    def __init__(self, X_train, y_train, X_test, y_test, X_val, y_val, batch_size=1):
        self.batch_size = batch_size
        self.num_classes = 10
        self.X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
        self.X_val = X_val.reshape(X_val.shape[0], X_val.shape[1] * X_val.shape[2])
        self.X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
        self.y_train = self.one_hot_encode(y_train)
        self.y_val = self.one_hot_encode(y_val)
        self.y_test = self.one_hot_encode(y_test)
        self.save_path = None

        # Network parameters
        input_size = self.X_train.shape[1]
        n_hidden = (input_size + self.num_classes) // 2  # Number of hidden layer neurons

        # Initialize network
        self.network = TwoLayerNN(input_size, n_hidden, self.num_classes)

    def one_hot_encode(self, y):
        """Transform the integer class labels into one-hot encoded arrays.

        :param y: an array of integers representing the class labels
        :return: an array of size (n_samples, n_classes)
        """
        one_hot_y = np.zeros((y.size, self.num_classes))
        one_hot_y[np.arange(y.size), y] = 1
        return one_hot_y

    @staticmethod
    def mse_loss(pred, true):
        """Calculates the mean squared error loss between the prediction and
        the ground truth labels.

        :param pred: an array of the probabilities of each class label
        :param true: an array of the one-hot encoded ground truth labels
        :return: a floating-point value representing the MSE loss
        """
        return np.mean(np.mean((true - pred) ** 2, axis=1))

    @staticmethod
    def get_predicted_class(pred):
        """Assigns a value of 1 to the column in each row of pred with the
        highest sigmoid logit value, 0 otherwise.

        :param pred: an array of the probabilities for each class label
        :return: an array of shape pred with the max value in each row given
                 a value of 1, 0 otherwise
        """
        results = np.zeros_like(pred)
        results[np.arange(len(pred)), pred.argmax(1)] = 1
        return results

    @staticmethod
    def get_accuracy(pred, true):
        """Calculates the accuracy of the prediction by dividing the number of
        correctly classified images by the total number of test samples.

        :param pred: an array of shape (n_samples, n_classes) of the predicted
                     class labels
        :param true: the ground truth class labels
        :return: a float representing the accuracy of the prediction
        """
        num_correct = np.sum((pred == true).all(axis=1))
        return num_correct / pred.shape[0]

    @staticmethod
    def get_confusion_matrix(pred, true):
        """Returns the confusion matrix for the predicted class labels.

        :param pred: an array of shape (n_samples, n_classes) of the prediction
        :param true: an array of shape (n_samples, n_classes) of ground truth
        :return: an array of shape (n_classes, n_classes) containing the
                 confusion matrix
        """
        return confusion_matrix(true.argmax(axis=1), pred.argmax(axis=1))

    @staticmethod
    def get_precision(cm):
        """Calculates the precision metric from an N-D confusion matrix.

        :param cm: an N-D array of the confusion matrix
        :return: the average precision across all class labels
        """
        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        return np.nanmean(tp / (tp + fp))

    @staticmethod
    def get_recall(cm):
        """Calculates the recall metric from an N-D confusion matrix.

        :param cm: an N-D array of the confusion matrix
        :return: the average recall across all class labels
        """
        tp = np.diag(cm)
        fn = np.sum(cm, axis=1) - tp
        return np.nanmean(tp / (tp + fn))

    def save_model(self, epoch, save_best=False):
        """Saves the current network's weights and biases as numpy arrays.

        :param epoch: a string used to identify the model checkpoint
        :param save_best: a bool indicating whether to save the best model
                          rather than the current one
        :return: None
        """
        if save_best:
            np.save(self.save_path + "/weights_jh_" + str(epoch) + ".npy", self.network.best_weights_jh)
            np.save(self.save_path + "/weights_hi_" + str(epoch) + ".npy", self.network.best_weights_hi)
            np.save(self.save_path + "/bias_h_" + str(epoch) + ".npy", self.network.best_bias_h)
            np.save(self.save_path + "/bias_i_" + str(epoch) + ".npy", self.network.best_bias_i)
        else:
            np.save(self.save_path + "/weights_jh_" + str(epoch) + ".npy", self.network.weights_jh)
            np.save(self.save_path + "/weights_hi_" + str(epoch) + ".npy", self.network.weights_hi)
            np.save(self.save_path + "/bias_h_" + str(epoch) + ".npy", self.network.bias_h)
            np.save(self.save_path + "/bias_i_" + str(epoch) + ".npy", self.network.bias_i)

    def save_confusion_matrix(self, cm, phase):
        """Generates and saves a plot of the given confusion matrix.

        :param cm: an array of the confusion matrix
        :param phase: a string to identify the phase in which the confusion
                      matrix was generated
        :return: None
        """
        classes = [str(digit) for digit in range(self.num_classes)]
        df_cm = pd.DataFrame(cm, index=classes, columns=classes)
        plt.figure(figsize=(15, 10))
        cm_plt = sn.heatmap(df_cm, annot=True, fmt="g")
        bottom, top = cm_plt.get_ylim()
        cm_plt.set_ylim(bottom + 0.5, top - 0.5)
        cm_plt.figure.savefig(self.save_path + "/" + phase + "_confusion_matrix.png")

    def train(self, epochs, min_delta=0, patience=3):
        """Trains the neural network and saves the weights at set intervals.

        :param epochs: the number of complete passes through the dataset
        :param min_delta: a float of the minimum change in the validation loss
                          to be considered an improvement
        :param patience: the number of epochs with no decrease in loss before
                         stopping the training early
        :return: None
        """
        # Create directory to save model checkpoints
        self.save_path = "saved_model_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.mkdir(self.save_path)
        # Create empty file to save model
        with open(self.save_path + "/model_checkpoints.csv", "w") as f:
            f.write("epoch,loss,accuracy,precision,recall,val_loss,val_accuracy,val_precision,val_recall\n")

        # Calculate number of batches to iterate over
        is_multiple = True if self.X_train.shape[0] % self.batch_size == 0 else False
        batches_per_epoch = self.X_train.shape[0] // self.batch_size
        batches_per_epoch = batches_per_epoch if is_multiple else batches_per_epoch + 1

        # Variables for early stopping
        previous_val_loss = None
        epochs_without_improvement = 0

        # Training loop
        for epoch in tqdm.tqdm(range(epochs)):
            # Shuffle training data
            p = rng.permutation(self.X_train.shape[0])
            self.X_train = self.X_train[p]
            self.y_train = self.y_train[p]

            # Initialize lists for storing loss/accuracy of each batch
            all_batch_loss = []
            all_batch_accuracy = []
            all_batch_precision = []
            all_batch_recall = []

            # Loop through number of batches
            with tqdm.tqdm(range(batches_per_epoch), position=0, leave=True) as pbar:
                for batch_idx in pbar:
                    # Prepare batches
                    if batch_idx == batches_per_epoch - 1 and not is_multiple:
                        # Add random samples from training set to fill in missing for last batch
                        x_batch = self.X_train[batch_idx * self.batch_size:]
                        y_batch = self.y_train[batch_idx * self.batch_size:]
                        num_missing = self.batch_size - len(self.X_train[batch_idx * self.batch_size:])
                        random_idx = rng.choice(range(self.X_train.shape[0]), size=num_missing, replace=False)
                        for idx in random_idx:
                            x_batch = np.concatenate((x_batch, np.expand_dims(self.X_train[idx], axis=0)))
                            y_batch = np.concatenate((y_batch, np.expand_dims(self.y_train[idx], axis=0)))
                    else:
                        x_batch = self.X_train[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
                        y_batch = self.y_train[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]

                    # Get prediction and calculate loss and metrics
                    pred = self.network.forward(x_batch)
                    loss = self.mse_loss(pred, y_batch)
                    accuracy = self.get_accuracy(self.get_predicted_class(pred), y_batch)
                    cm = self.get_confusion_matrix(pred, y_batch)
                    precision = self.get_precision(cm)
                    recall = self.get_recall(cm)
                    all_batch_loss.append(loss)
                    all_batch_accuracy.append(accuracy)
                    all_batch_precision.append(precision)
                    all_batch_recall.append(recall)
                    pbar.set_description(f"loss: {loss:.4f} accuracy: {accuracy:.4f}")

                    # Gradient descent
                    self.network.backprop(y_batch)

            # Display and save train/validation loss/metrics after each epoch
            epoch_loss = sum(all_batch_loss) / len(all_batch_loss)
            epoch_acc = sum(all_batch_accuracy) / len(all_batch_accuracy)
            epoch_precision = sum(all_batch_precision) / len(all_batch_precision)
            epoch_recall = sum(all_batch_recall) / len(all_batch_recall)
            val_pred = self.network.forward(self.X_val)
            val_loss = self.mse_loss(val_pred, self.y_val)
            val_acc = self.get_accuracy(self.get_predicted_class(val_pred), self.y_val)
            val_cm = self.get_confusion_matrix(val_pred, self.y_val)
            val_precision = self.get_precision(val_cm)
            val_recall = self.get_recall(val_cm)
            print(f"Epoch {epoch}: loss: {epoch_loss:.4f} "
                  f"accuracy: {epoch_acc:.4f} "
                  f"precision: {epoch_precision:.4f} "
                  f"recall: {epoch_recall:.4f} "
                  f"val_loss: {val_loss:.4f} "
                  f"val_acc: {val_acc:.4f} "
                  f"val_precision: {val_precision:.4f} "
                  f"val_recall: {val_recall:.4f}")
            with open(self.save_path + "/model_checkpoints.csv", "a") as f:
                f.write(f"{epoch},{epoch_loss},{epoch_acc},{epoch_precision},{epoch_recall},"
                        f"{val_loss},{val_acc},{val_precision},{val_recall}\n")

            # Save weights/biases every 25 epochs
            if epoch % 25 == 0:
                self.save_model(epoch)

            # Update variables for early stopping
            if previous_val_loss:
                if val_loss - previous_val_loss >= min_delta:
                    epochs_without_improvement += 1
                    if epochs_without_improvement == patience:
                        # Quit training and save
                        print(f"Validation loss has not decreased for {patience} epochs. Ending training.")
                        self.save_model("best_model", save_best=True)
                        self.save_confusion_matrix(val_cm, "train")
                        break
                # Reset counter if loss has decreased
                else:
                    self.network.set_best_weights()
                    epochs_without_improvement = 0
            previous_val_loss = val_loss

        # Save confusion matrix
        val_pred = self.network.forward(self.X_val)
        val_cm = self.get_confusion_matrix(val_pred, self.y_val)
        self.save_confusion_matrix(val_cm, "train")

    def test(self, model_path=None):
        """Tests the trained model on the test dataset.

        :param model_path: the path to the saved model
        :return: None
        """
        # Load model weights and biases
        if model_path:
            self.save_path = model_path
            self.network.weights_hi = np.load(model_path + "/weights_hi_best_model.npy")
            self.network.weights_jh = np.load(model_path + "/weights_jh_best_model.npy")
            self.network.bias_i = np.load(model_path + "/bias_i_best_model.npy")
            self.network.bias_h = np.load(model_path + "/bias_h_best_model.npy")

        # Get prediction and calculate metrics
        pred = self.network.forward(self.X_test)
        accuracy = self.get_accuracy(self.get_predicted_class(pred), self.y_test)
        cm = self.get_confusion_matrix(pred, self.y_test)
        precision = self.get_precision(cm)
        recall = self.get_recall(cm)
        print(f"Results of {self.save_path} on test data:\n"
              f"\tAccuracy: {accuracy:.4f}\n"
              f"\tPrecision: {precision:.4f}\n"
              f"\tRecall: {recall:.4f}")
        self.save_confusion_matrix(cm, "test")


if __name__ == '__main__':
    # Load the data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Leave 10000 samples for validation
    X_train, X_val = X_train[:50000], X_train[50000:]
    y_train, y_val = y_train[:50000], y_train[50000:]

    # Initialize model
    model = Model(X_train, y_train, X_test, y_test, X_val, y_val, batch_size=64)
    model.train(1000)
    model.test()
