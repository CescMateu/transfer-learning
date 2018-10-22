"""Define the model."""

import tensorflow as tf

from model.cnn import CNN


def build_model(input_tensor, params, is_training, reuse):
    '''Creates the corresponding TensorFlow graph that needs to be executed inside a tf.Session().

    :param input_x
    :param labels
    :param dropout_rate
    :param learning_rate

    :return: Compute logits of the model (output distribution)
    '''

    # Feeding the 'inputs' data to the model
    cnn = CNN()
    cnn.build(input_tensor, params, is_training, reuse)
    logits = cnn.logits

    print(cnn.bn1)
    print(cnn.conv4)

    return logits


def model_fn(inputs, params, is_training):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """

    # Extract the inputs and cast them into the appropriate data types
    labels = tf.cast(inputs['labels'], tf.int32)
    images = tf.cast(inputs['images'], tf.float32)

    # Set the 'reuse' parameter from tf.variable_scope() correctly
    reuse = not is_training

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    # Compute the output distribution of the model and the predictions
    logits = build_model(images, params, is_training=is_training, reuse=reuse)
    predictions = tf.cast(tf.argmax(logits, 1), tf.int32)

    # Define loss and accuracy
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy_vec = tf.cast(tf.equal(labels, predictions), tf.float32)
    accuracy = tf.reduce_mean(accuracy_vec)

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.cast(tf.argmax(logits, 1), tf.int32)),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('train_image', inputs['images'])

    # Add incorrectly labeled images
    mask = tf.not_equal(labels, predictions)

    # Add a different summary to know how they were misclassified
    for label in range(0, params.num_labels):
        mask_label = tf.logical_and(mask, tf.equal(predictions, label))
        incorrect_image_label = tf.boolean_mask(inputs['images'], mask_label)
        tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
