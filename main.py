print('Importing libraries...')
# PyQT (GUI Library)
from PyQt5.QtWidgets import (QLabel, QFormLayout, QLineEdit, QSizePolicy, QFileDialog, QGroupBox)
from PyQt5.QtCore import (QThread, QObject, pyqtSignal, pyqtSlot)
from PyQt5 import QtWidgets
from form_mainWindow import Ui_MainWindow

# Machine Learning Libraries
import tensorflow as tf
import numpy as np
import scipy.io as sio
import random
import os
import sys
import sklearn.metrics as sk

from Timer import Timer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warning messages

class Worker(QObject): # Thread
    def __init__(self, hidden_layers, drop_keep_prob, batch_size, num_steps, filepath):
        QObject.__init__(self)
        self.hidden_layers = hidden_layers
        self.drop_keep_prob = drop_keep_prob
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.filepath = filepath
    finished = pyqtSignal()
    log = pyqtSignal(str)
    @pyqtSlot()
    def beginLearning(self):
        print_step_offset = self.num_steps // 10
        # beta = 0.001 # Use this for L2 Regularization

        ## Load Feature Vector from file
        self.log.emit('\nLoading saved features from file......\n')
        featureVector = sio.loadmat(self.filepath)
        num_labels = len(np.unique(featureVector['y_valid']))  # Change this if changing the dataset

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

        def remove_random(l, mirror, count, label):
            indices = [i for i, x in enumerate(l) if x == label]
            indices = random.sample(indices, count)
            return np.delete(l, indices, axis=0), np.delete(mirror, indices, axis=0)

        ## Balance the dataset (remove exceeding samples)
        _, number_of_samples = np.unique(featureVector['y_train'], return_counts=True)
        min_of_number_of_samples = np.min(number_of_samples)
        for i in range(0,num_labels):
            if number_of_samples[i] - min_of_number_of_samples > 0:
                featureVector['y_train'], featureVector['X_train'] = remove_random(featureVector['y_train'], featureVector['X_train'], number_of_samples[i] - min_of_number_of_samples, i+1)

        _, number_of_samples = np.unique(featureVector['y_valid'], return_counts=True)
        min_of_number_of_samples = np.min(number_of_samples)
        for i in range(0, num_labels):
            if number_of_samples[i] - min_of_number_of_samples > 0:
                featureVector['y_valid'], featureVector['X_valid'] = remove_random(featureVector['y_valid'], featureVector['X_valid'], number_of_samples[i] - min_of_number_of_samples, i + 1)

        # featureVector['y_train'], featureVector['X_train'] = remove_random(featureVector['y_train'], featureVector['X_train'], number_of_samples[0]-min_of_number_of_samples, 1)
        # featureVector['y_train'], featureVector['X_train'] = remove_random(featureVector['y_train'], featureVector['X_train'], number_of_samples[1]-min_of_number_of_samples, 2)
        # featureVector['y_train'], featureVector['X_train'] = remove_random(featureVector['y_train'], featureVector['X_train'], number_of_samples[2]-min_of_number_of_samples, 3)

        _, number_of_samples2_ = np.unique(featureVector['y_train'], return_counts=True)
        print(number_of_samples2_)
        featureVector['y_valid'] = one_hot(featureVector['y_valid'], True)
        featureVector['y_train'] = one_hot(featureVector['y_train'], True)
        featureVector['y_test'] = one_hot(featureVector['y_test'], True)

        featureVector['X_valid'] = featureVector['X_valid'].astype(np.float32)
        featureVector['X_train'] = featureVector['X_train'].astype(np.float32)
        featureVector['X_test'] = featureVector['X_test'].astype(np.float32)

        ## Model definition
        input_size = featureVector['X_test'].shape[1]
        num_hidden_layers = len(self.hidden_layers)

        graph = tf.Graph()
        with graph.as_default():
            ## Input data. For the training data, we use a placeholder that will be fed
            ## at run time with a training minibatch.
            tf_train_dataset = tf.placeholder(tf.float32,
                                              shape=(self.batch_size, input_size))
            tf_train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, num_labels))
            tf_valid_dataset = tf.constant(featureVector['X_valid'])
            tf_test_dataset = tf.constant(featureVector['X_test'])
            keep_prob = tf.placeholder(tf.float32, shape=(num_hidden_layers))

            ## Initialize weights and biases
            W = []
            b = []
            for i in range(0, num_hidden_layers + 1):
                if i == 0:
                    W.append(tf.Variable(tf.truncated_normal([input_size, self.hidden_layers[i]])))
                    b.append(tf.Variable(tf.zeros([self.hidden_layers[i]])))
                elif i == num_hidden_layers:
                    W.append(tf.Variable(tf.truncated_normal([self.hidden_layers[i - 1], num_labels])))
                    b.append(tf.Variable(tf.zeros([num_labels])))
                else:
                    W.append(tf.Variable(tf.truncated_normal([self.hidden_layers[i - 1], self.hidden_layers[i]])))
                    b.append(tf.Variable(tf.zeros([self.hidden_layers[i]])))

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
            # optimizer = tf.train.GradientDescentOptimizer(0.3).minimize(loss)
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
            # print("Model initialized")
            self.log.emit('Model initialized')
            for step in range(self.num_steps+1):
                # #Pick an offset within the training data, which has been randomized.
                ## Note: we could use better randomization across epochs.
                offset = (step * self.batch_size) % (featureVector['y_train'].shape[0] - self.batch_size)
                # Generate a minibatch.
                batch_data = featureVector['X_train'][offset:(offset + self.batch_size), :]
                batch_labels = featureVector['y_train'][offset:(offset + self.batch_size), :]

                ## Prepare a dictionary telling the session where to feed the minibatch.
                ## The key of the dictionary is the placeholder node of the graph to be fed,
                ## and the value is the numpy array to feed to it.
                _, l, predictions = session.run(
                    [optimizer, loss, train_prediction],
                    feed_dict={tf_train_dataset: batch_data, tf_train_labels: batch_labels
                               # , keep_prob1 : drop_keep_prob1, keep_prob2 : drop_keep_prob2, keep_prob3 : drop_keep_prob3, keep_prob4 : drop_keep_prob3
                        , keep_prob: self.drop_keep_prob
                               }
                )
                if (step % print_step_offset == 0):
                    self.log.emit("\nMinibatch loss at step %d: %f" % (step, l))
                    self.log.emit("\nMinibatch accuracy: %.1f%%" % accuracy(predictions=predictions, labels=batch_labels))

            pred_valid = valid_prediction.eval(feed_dict={keep_prob: [1] * num_hidden_layers})
            y_pred = one_hot(pred_valid, False).eval()
            y_true = one_hot(featureVector['y_valid'], False).eval()

            self.log.emit('\n\n----- RESULTS -----')
            self.log.emit("\nValidation Accuracy: %.3f" % sk.accuracy_score(y_pred=y_pred, y_true=y_true))
            self.log.emit("\n\nValidation Sensitivity (Recall): %s" % str(sk.recall_score(y_pred=y_pred, y_true=y_true, average=None)))
            self.log.emit("\n\nValidation Positive Predictive Value (Precision): %s" % str(sk.precision_score(y_pred=y_pred, y_true=y_true, average=None)))
            self.log.emit("\nValidation Confusion Matrix:\n")
            self.log.emit(str(sk.confusion_matrix(y_pred=y_pred, y_true=y_true)))

        self.finished.emit()

class MyForm(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.beginLearning_pushButton.clicked.connect(self.begin_learning)
        self.ui.browse_pushButton.clicked.connect(self.browseFile)
        self.ui.clearLog_pushButton.clicked.connect(self.clearLog)
        self.ui.saveLog_pushButton.clicked.connect(self.saveLog)

        self.ui.addHiddenLayer_pushButton.clicked.connect(self.addHiddenLayer)
        self.num_hidden_layers = 1
        self.addHiddenLayer()

        self.thread = QThread()


    def browseFile(self):
        try:
            filepath, _ = QFileDialog.getOpenFileName(self, 'Select Dataset', ".", 'Dataset Files (*.mat)')
            self.ui.filepath_lineEdit.setText(filepath)
        except:
            pass


    def saveLog(self):
        try:
            fileName,_ = QFileDialog.getSaveFileName(self, 'Save Log File', '.', 'Text file (*.txt)')
            with open(fileName, 'w') as f:
                f.write(self.ui.log_plainTextEdit.toPlainText())
        except:
            pass


    def clearLog(self):
        self.ui.log_plainTextEdit.clear()



    def addHiddenLayer(self):
        self.newWidget = QGroupBox("Hidden Layer %d" % self.num_hidden_layers)
        self.newWidget.setLayout(QFormLayout())

        self.newWidget.layout().addWidget(QLabel("Neurons"))
        neurons_lineEdit = QLineEdit()
        neurons_lineEdit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        neurons_lineEdit.setObjectName("neurons_%d_lineEdit" % self.num_hidden_layers)
        self.newWidget.layout().addWidget(neurons_lineEdit)

        self.newWidget.layout().addWidget(QLabel("Dropout"))
        dropout_lineEdit = QLineEdit()
        dropout_lineEdit.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        dropout_lineEdit.setObjectName("dropout_%d_lineEdit" % self.num_hidden_layers)
        self.newWidget.layout().addWidget(dropout_lineEdit)

        self.ui.scrollAreaWidgetContents.layout().addWidget(self.newWidget)
        self.num_hidden_layers += 1


    def begin_learning(self):

        batch_size = int(self.ui.batchSize_lineEdit.text())
        num_steps = int(self.ui.steps_lineEdit.text())

        hidden_layers = []
        for i in range(1, self.num_hidden_layers):
            object = self.findChild(QtWidgets.QLineEdit, "neurons_%d_lineEdit" % i)
            hidden_layers.append(int(object.text()))

        drop_keep_prob = []
        for i in range(1, self.num_hidden_layers):
            object = self.findChild(QtWidgets.QLineEdit, "dropout_%d_lineEdit" % i)
            drop_keep_prob.append(float(object.text()))

        self.obj = Worker(batch_size = batch_size, num_steps = num_steps, hidden_layers = hidden_layers, drop_keep_prob = drop_keep_prob, filepath=str(self.ui.filepath_lineEdit.text()))
        self.obj.log.connect(self.printToLog)
        self.obj.moveToThread(self.thread)
        self.obj.finished.connect(self.learning_finished)
        self.thread.started.connect(self.obj.beginLearning)
        self.ui.beginLearning_pushButton.setEnabled(False)
        self.thread.start()

    def printToLog(self, text):
        self.ui.log_plainTextEdit.insertPlainText(text)
        self.ui.log_plainTextEdit.ensureCursorVisible()

    def learning_finished(self):
        self.thread.quit()
        self.ui.beginLearning_pushButton.setEnabled(True)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = MyForm()
    myapp.show()
    sys.exit(app.exec_())
