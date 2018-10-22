import tensorflow as tf

from model.model_template import ModelTemplate


class CNN(ModelTemplate):

    def __init__(self):

        # Inputs
        self.data = None
        # Block 1
        self.conv1 = None
        self.bn1 = None
        self.relu1 = None
        self.pool1 = None
        # Block 2
        self.conv2 = None
        self.bn2 = None
        self.relu2 = None
        self.pool2 = None
        # Block 3
        self.conv3 = None
        self.bn3 = None
        self.relu3 = None
        self.pool3 = None
        # Block 4
        self.conv4 = None
        self.bn4 = None
        self.relu4 = None
        self.pool4 = None
        # FC 1
        self.flatten1 = None
        self.fc1 = None
        self.bn5 = None
        self.relu5 = None
        # FC2
        self.logits = None

    def build(self, input_tensor, params, is_training, reuse):

        if input_tensor is None:
            raise ValueError('"input_tensor" must be a valid tf.Tensor.')

        assert input_tensor.get_shape().as_list() == [None, params.image_size, params.image_size, 3]

        # Save the input tensor
        self.data = input_tensor

        # Extract parameters from the input
        num_channels = params.num_channels
        bn_momentum = params.bn_momentum
        num_labels = params.num_labels

        ### CNN Architecture ###

        # We will use several blocks of (CONV 3x3 - BATCH NORM - RELU -  MAXPOOL 2x2) with different number of
        # channels in each one. We will also use Batch Normalization, which will help us to speed us the learning and
        # also allows each layer of the network to learn by itself a little bit more independently of other layers. What
        # it does is normalizes the output of the previous layer.

        # Layer 0 (Input) -> Output size: (None, 64, 64, 3)

        with tf.variable_scope('model', reuse=reuse):

            ######################################################
            # Block 1 (CONV 3x3 - BATCH NORM - RELU -  MAXPOOL 2x2)

            conv1 = tf.layers.conv2d(inputs=input_tensor, filters=num_channels,
                                     kernel_size=3, padding='same', name='conv1')
            self.conv1 = conv1

            if params.use_batch_norm:
                conv1 = tf.layers.batch_normalization(inputs=conv1, momentum=bn_momentum,
                                                      training=is_training, name='bn1')
                self.bn1 = conv1

            relu1 = tf.nn.relu(conv1, name='relu1')
            pool1 = tf.layers.max_pooling2d(inputs=relu1, pool_size=2, strides=2,
                                            padding='valid', name='pool1')
            self.relu1 = relu1
            self.pool1 = pool1

            # Output size (32, 32, num_channels)
            assert pool1.get_shape().as_list() == [None, 32, 32, num_channels]

            ######################################################
            # Block 2 (CONV 3x3 - BATCH NORM - RELU -  MAXPOOL 2x2)

            conv2 = tf.layers.conv2d(inputs=pool1, filters=num_channels*2,
                                     kernel_size=3, padding='same', name='conv2')
            self.conv2 = conv2

            if params.use_batch_norm:
                conv2 = tf.layers.batch_normalization(inputs=conv2, momentum=bn_momentum,
                                                      training=is_training, name='bn2')
                self.bn2 = conv2

            relu2 = tf.nn.relu(conv2)
            pool2 = tf.layers.max_pooling2d(inputs=relu2, pool_size=2,
                                            strides=2, padding='valid', name='pool2')
            self.relu2 = relu2
            self.pool2 = pool2

            # Output size (16, 16, num_channels*2)
            assert pool2.get_shape().as_list() == [None, 16, 16, num_channels*2]

            ######################################################
            # Block 3 (CONV 3x3 - BATCH NORM - RELU -  MAXPOOL 2x2)

            conv3 = tf.layers.conv2d(inputs=pool2, filters=num_channels * 3,
                                     kernel_size=3, padding='same', name='conv3')
            self.conv3 = conv3

            if params.use_batch_norm:
                conv3 = tf.layers.batch_normalization(inputs=conv3, momentum=bn_momentum,
                                                      training=is_training, name='bn3')
                self.bn3 = conv3

            relu3 = tf.nn.relu(conv3, name='relu3')
            pool3 = tf.layers.max_pooling2d(inputs=relu3, pool_size=2, strides=2,
                                            padding='valid', name='pool3')
            self.relu3 = relu3
            self.pool3 = pool3

            # Output size (8, 8, num_channels*3)
            assert pool3.get_shape().as_list() == [None, 8, 8, num_channels * 3]

            ######################################################
            # Block 4 (CONV 3x3 - BATCH NORM - RELU -  MAXPOOL 2x2)

            conv4 = tf.layers.conv2d(inputs=pool3, filters=num_channels * 4,
                                     kernel_size=3, padding='same', name='conv4')
            self.conv4 = conv4

            if params.use_batch_norm:
                conv4 = tf.layers.batch_normalization(inputs=conv4, momentum=bn_momentum,
                                                      training=is_training, name='bn4')
                self.bn4 = conv4

            relu4 = tf.nn.relu(conv4, name='relu4')
            pool4 = tf.layers.max_pooling2d(inputs=relu4, pool_size=2,
                                            strides=2, padding='valid', name='pool4')
            self.relu4 = relu4
            self.pool4 = pool4

            # Output size (4, 4, num_channels*4)
            assert pool4.get_shape().as_list() == [None, 4, 4, num_channels * 4]

            ######################################################
            # Block 5 (FC w/ num_channels*8 output - BATCH NORM - RELU)

            # We need to flatten our CNN at this point, and for doing so we need a tf.reshape instead
            # of a tf.layers.flatten. tf.reshape has a static shape, meaning that some of the sizes are
            # independent of the rest of the network. tf.layers.dense needs that the last size of the previous
            # layer to be known.


            # Flatten the output result of the last block for building a FC layer
            flatten1 = tf.reshape(pool4, [-1, 4 * 4 * num_channels * 4], name='flatten1')
            self.flatten1 = flatten1

            fc1 = tf.layers.dense(inputs=flatten1, units=num_channels*8, name='fc1')
            self.fc1 = fc1

            if params.use_batch_norm:
                fc1 = tf.layers.batch_normalization(inputs=fc1, momentum=bn_momentum,
                                                    training=is_training, name='bn5')
                self.bn5 = fc1

            relu5 = tf.nn.relu(fc1, name='relu5')
            self.relu5 = relu5

            # Output size (None, num_channels * 8)
            assert relu5.get_shape().as_list() == [None, num_channels * 8]

            ######################################################
            # Block 6 (FC w/ num_labels output - BATCH NORM - RELU)

            logits = tf.layers.dense(inputs=relu5, units=num_labels, name='logits')

            assert logits.get_shape().as_list() == [None, num_labels]

            self.logits = logits