import model_template
import tensorflow as tf


class CNN(model_template.ModelTemplate):

    def __init__(self, dropout_rate):

        # Inputs
        self.dropout_rate = dropout_rate
        self.data = None

        # Embeddings
        self.conv1 = None
        self.pool1 = None
        self.conv2 = None
        self.pool2 = None
        self.flatten = None
        self.fc1 = None

        # Output
        self.logits = None

    def build(self, input_tensor):

        if input_tensor is None:
            input_tensor = tf.placeholder(tf.float32, shape=(None, 64, 64, 3), name='data')

        self.data = input_tensor
        # Layer 1 (CONV-MAXPOOL) -> Output size: 28x28x15
        conv1 = tf.layers.conv2d(input_tensor, 15, 9, activation=tf.nn.relu, name='conv1')
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2, name="pool1")

        # Layer 2 (CONV-MAXPOOL) -> Output size: 12x12x25
        conv2 = tf.layers.conv2d(pool1, 25, 4, activation=tf.nn.relu, name='conv2')
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2, name="pool2")

        # Layer 2 (FC without dropout)
        # We need to flatten our CNN at this point, and for doing so we need a tf.reshape instead
        # of a tf.layers.flatten. tf.reshape has a static shape, meaning that some of the sizes are
        # independent of the rest of the network. tf.layers.dense needs that the last size of the previous
        # layer to be known.
        flatten = tf.reshape(pool2, [-1, 12*12*25], name='flatten')
        fc1 = tf.layers.dense(flatten, (12*12*25)/2, activation=tf.nn.relu, name='fc1')

        logits = tf.layers.dense(fc1, 6, activation=None, name='logits')

        self.conv1 = conv1
        self.pool1 = pool1
        self.conv2 = conv2
        self.pool2 = pool2
        self.flatten = flatten
        self.fc1 = fc1
        self.logits = logits
