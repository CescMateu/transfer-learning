from aux import *
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

####################################

# Global variables
EPOCHS = 100
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.80
BATCH_SIZE = 1080

# Local variables
print_loss = True

####################################

# Load the SIGNS data
x_train, y_train, x_valid, y_valid, classes = load_dataset('data/')

# Transform the labels to a one-hot encoding format
y_train_one_hot = one_hot_encoding(y_train, depth=6)
y_valid_one_hot = one_hot_encoding(y_valid, depth=6)

# Create placeholders for the (X, y) pairs
input_x, labels = create_placeholders()

# Create a train tf.Dataset and an iterator for extracting the data
train_dataset = create_tf_dataset(input_x, labels, BATCH_SIZE)
iterator = train_dataset.make_initializable_iterator()  # Create the iterator
batch = iterator.get_next()

# Create TensorFlow graph and check the returned objects
optimizer, loss, cnn, accuracy = create_graph(batch[0], batch[1], DROPOUT_RATE, LEARNING_RATE)


# Initialize the tf.Session() and run the graph
with tf.Session() as sess:

    # Initialize global/local variables
    sess.run(tf.global_variables_initializer())
    #sess.run(tf.local_variables_initializer())

    # Initialize the iterator with training data
    sess.run(iterator.initializer, feed_dict={input_x: x_train, labels: y_train_one_hot})

    ### Training mode ###
    losses = []
    for i in range(EPOCHS):
        _, l, acc = sess.run([optimizer, loss, accuracy])

        if i % 5 == 0:
            print("Epoch: {}, loss: {:.3f}, training accuracy: {:.3f}%".format(i, l, acc))
        if print_loss:
            losses.append(l)
    print("Epoch: {}, loss: {:.3f}, training accuracy: {:.3f}%".format(i, l, acc))

    if print_loss:
        # Plot the cost
        plt.plot(np.squeeze(losses))
        plt.ylabel('Loss function')
        plt.xlabel('Iterations (per tens)')
        plt.title("Learning rate =" + str(LEARNING_RATE))
        plt.show()

    ### Validation mode ###
    # Initialize the iterator with validation data
    sess.run(iterator.initializer, feed_dict={input_x: x_valid, labels: y_valid_one_hot})

    # Initialize variables
    avg_acc = 0
    valid_iterations = 50
    for i in range(valid_iterations):
        acc = sess.run([accuracy])
        avg_acc += acc[0]

    avg_acc = (avg_acc/valid_iterations)
    print("Average validation set accuracy over {} iterations is {:.3f}%".format(valid_iterations, avg_acc))