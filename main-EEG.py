print('Importing libraries...')
import tensorflow as tf
import numpy as np
import scipy.io as sio
import random
import math
import os
import sklearn.metrics as sk

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warning messages

## CHANGE THESE VALUES TO YOUR NEEDS

batch_size = 128
hidden_layers = [3200, 1600, 3200, 800, 3200, 400]
drop_keep_prob = [0.1, 0.5, 0.3, 0.5, 0.7, 0.9]  # MUST have the same number of items as hidden_layers. Change all to 1 if Dropout shouldn't be used
num_steps = 2001

print_step_offset = 1000
# beta = 0.001 # Use this for L2 Regularization

num_labels = 3  # Change this if changing the dataset

## Load Feature Vector from file
print('\nLoading saved features from file......\n')
featureVector = sio.loadmat('featuresdata_eegmmidb.mat')


def one_hot(labels, on=True):
    # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
    if on:
        labels = (np.arange(num_labels) == labels[:, None] - 1).astype(np.float32)
        labels = np.array(labels.reshape((labels.shape[0], labels.shape[2])))
        return labels
    else:
        return tf.argmax(labels, axis=1)


def accuracy(predictions, labels):  # Defines the accuracy method
    return (100.0 * np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))
            / predictions.shape[0])


def remove_random(l, mirror, count):
    indices = [i for i, x in enumerate(l) if x == 1]
    indices = random.sample(indices, count)
    return np.delete(l, indices, axis=0), np.delete(mirror, indices, axis=0)


featureVector['y_train'], featureVector['X_train'] = remove_random(featureVector['y_train'], featureVector['X_train'], 140000)

featureVector['y_valid'] = one_hot(featureVector['y_valid'], True)
featureVector['y_train'] = one_hot(featureVector['y_train'], True)
featureVector['y_test'] = one_hot(featureVector['y_test'], True)

featureVector['X_valid'] = featureVector['X_valid'].astype(np.float32)
featureVector['X_train'] = featureVector['X_train'].astype(np.float32)
featureVector['X_test'] = featureVector['X_test'].astype(np.float32)

## Model definition
input_size = featureVector['X_test'].shape[1]
num_hidden_layers = len(hidden_layers)

graph = tf.Graph()
with graph.as_default():
    ## Input data. For the training data, we use a placeholder that will be fed
    ## at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, input_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(featureVector['X_valid'])
    tf_test_dataset = tf.constant(featureVector['X_test'])
    keep_prob = tf.placeholder(tf.float32, shape=(num_hidden_layers))

    ## Initialize weights and biases
    W = []
    b = []
    for i in range(0, num_hidden_layers + 1):
        if i == 0:
            W.append(tf.Variable(tf.truncated_normal([input_size, hidden_layers[i]])))
            b.append(tf.Variable(tf.zeros([hidden_layers[i]])))
        elif i == num_hidden_layers:
            W.append(tf.Variable(tf.truncated_normal([hidden_layers[i - 1], num_labels])))
            b.append(tf.Variable(tf.zeros([num_labels])))
        else:
            W.append(tf.Variable(tf.truncated_normal([hidden_layers[i - 1], hidden_layers[i]])))
            b.append(tf.Variable(tf.zeros([hidden_layers[i]])))

    ## Training computation with dropout implemented

    temp_h_out = tf.nn.dropout(tf.nn.relu(tf.matmul(tf_train_dataset, W[0]) + b[0]), keep_prob[0])
    for i in range(1, num_hidden_layers):
        temp_h_out = tf.nn.dropout(tf.nn.relu(tf.matmul(temp_h_out, W[i]) + b[i]), keep_prob[i])
    logits = tf.matmul(temp_h_out, W[-1]) + b[-1]

    ## L2 Regularization
    # tf_vars   = tf.trainable_variables()
    # lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars])*beta
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))  # + lossL2

    ## Optimizer.
    # optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    ## Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)

    valid_prediction = tf.nn.relu(tf.matmul(tf_valid_dataset, W[0]) + b[0])
    for i in range(1, num_hidden_layers):
        valid_prediction = tf.nn.relu(tf.matmul(valid_prediction, W[i]) + b[i])
    valid_prediction = tf.nn.softmax(tf.matmul(valid_prediction, W[-1]) + b[-1])

    test_prediction = tf.nn.relu(tf.matmul(tf_test_dataset, W[0]) + b[0])
    for i in range(1, num_hidden_layers):
        test_prediction = tf.nn.relu(tf.matmul(test_prediction, W[i]) + b[i])
    test_prediction = tf.nn.softmax(tf.matmul(test_prediction, W[-1]) + b[-1])

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Model initialized")
    for step in range(num_steps):
        # #Pick an offset within the training data, which has been randomized.
        ## Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (featureVector['y_train'].shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = featureVector['X_train'][offset:(offset + batch_size), :]
        batch_labels = featureVector['y_train'][offset:(offset + batch_size), :]

        ## Prepare a dictionary telling the session where to feed the minibatch.
        ## The key of the dictionary is the placeholder node of the graph to be fed,
        ## and the value is the numpy array to feed to it.
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction],
            feed_dict={tf_train_dataset: batch_data, tf_train_labels: batch_labels
                       # , keep_prob1 : drop_keep_prob1, keep_prob2 : drop_keep_prob2, keep_prob3 : drop_keep_prob3, keep_prob4 : drop_keep_prob3
                , keep_prob: drop_keep_prob
                       }
        )
        if (step % print_step_offset == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions=predictions, labels=batch_labels))

    pred_valid = valid_prediction.eval(feed_dict={keep_prob: [1] * num_hidden_layers})
    y_pred = one_hot(pred_valid, False).eval()
    y_true = one_hot(featureVector['y_valid'], False).eval()

    print('\n\n----- RESULTS -----')
    print("\nValidation Accuracy: %.3f" % sk.accuracy_score(y_pred=y_pred, y_true=y_true))
    print("\nValidation Sensitivity (Recall):")
    print(sk.recall_score(y_pred=y_pred, y_true=y_true, average=None))
    print("\nValidation Positive Predictive Value (Precision):")
    print(sk.precision_score(y_pred=y_pred, y_true=y_true, average=None))
    print("\nValidation Confusion Matrix:")
    print(sk.confusion_matrix(y_pred=y_pred, y_true=y_true))
