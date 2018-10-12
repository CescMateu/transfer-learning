import h5py
import tensorflow as tf
import numpy as np
from cnn import *


def one_hot_encoding(a, depth):
    '''
    Transforms a one-dimensional vector with 'depth' unique values into a one-hot matrix
    :param a: One-dimensional vector containing the labels
    :param depth: Number of unique labels

    :return: Array with the corresponding one-hot encoding of size (|a|, depth)
    '''

    one_hot = np.zeros((a.size, depth), dtype=int)
    one_hot[np.arange(a.size), a] = 1

    return one_hot


def load_dataset(data_path):
    '''
    Loads the SIGNS dataset used during the Deep Learning Online Specialization from Coursera.

    :return: 5 objects containing the training and testing datasets
    '''

    # Read the data from the fole
    train_dataset = h5py.File(str(data_path) + 'signs/train_signs.h5', "r")
    test_dataset = h5py.File(str(data_path) + 'signs/test_signs.h5', "r")

    # Load the training and testing data in the correct format
    x_train = np.array(train_dataset["train_set_x"][:])  # your train set features
    y_train = np.array(train_dataset["train_set_y"][:])  # your train set labels
    x_test = np.array(test_dataset["test_set_x"][:])  # your test set features
    y_test = np.array(test_dataset["test_set_y"][:])  # your test set labels
    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    # Reshape the datasets in the correct format
    y_train = y_train.reshape(y_train.shape[0])
    y_test = y_test.reshape(y_test.shape[0])

    return x_train, y_train, x_test, y_test, classes

def create_placeholders():
    '''
    Create the corresponding placeholders for feeding data to the model

    :return: 4 objects, corresponding to the placeholders for the (x,y) data, the dropout rate and the learning rate
    '''

    input_x = tf.placeholder(tf.float32, shape=(None, 64, 64, 3), name='input_x')
    labels = tf.placeholder(tf.int32, shape=(None, 6), name='labels')

    return input_x, labels

def create_tf_dataset(input_x, labels, batch_size):
    '''
    Creates a (X, y) paired tf.data.Dataset for feeding into the models

    :param input_x: X matrix
    :param labels: y values
    :param batch_size: Size of the batches that will feed the model

    :return: A tf.data.Dataset object
    '''

    x = tf.data.Dataset.from_tensor_slices(input_x)
    y = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((x, y)).repeat().batch(batch_size).shuffle(500)

    return dataset

def create_graph(input_x, labels, dropout_rate = 0.7, learning_rate = 0.01):
    '''
    Creates the corresponding TensorFlow graph that needs to be executed inside a tf.Session().

    :param input_x
    :param labels
    :param dropout_rate
    :param learning_rate

    :return: Two nodes of the tf.graph corresponding to the cost and the optimizer.
    '''

    # Feeding the data to the model
    with tf.name_scope("model"):
        cnn = CNN_arq(dropout_rate)
        cnn.build(input_x)
        logits = cnn.logits

    # Compute the loss with the predictions and the correct labels
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='loss'))

    # Compute the accuracy
    predictions = tf.argmax(logits, 1, name='predictions')
    real_labels = tf.argmax(labels, 1, name = 'real_labels')
    correct_predictions = tf.equal(predictions, real_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

    # Choose the optimizer and the learning rate
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer').minimize(loss)

    return optimizer, loss, cnn, accuracy

