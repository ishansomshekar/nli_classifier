import csv
import gzip
import heapq
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import pickle
import time

import numpy as np
from scipy.signal import argrelextrema
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.platform import gfile

from build_embedding import EmbeddingWrapper
import config
from utils import pad_sequences, make_batches
from progbar import Progbar

logger = logging.getLogger("hw3.q2")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

t = time.localtime()
timeString  = time.strftime("%Y%m%d%H%M%S", t)
train_name = "baseline_" + str(time.time())
logs_path = os.getcwd() + '/tf_log/'
train = False
train_datapath = os.path.join(config.DATA_DIR, 'speech_transcriptions/train/tokenized/')
label_data = os.path.join(config.DATA_DIR, 'labels/train/labels.train.csv')

lang_dict = {
    'HIN' : 0,
    'ARA' : 1,
    'JPN' : 2,
    'SPA' : 3,
    'TUR' : 4,
    'GER' : 5,
    'TEL' : 6,
    'KOR' : 7,
    'ITA' : 8,
    'CHI' : 9,
    'FRE' : 10
}


class BaselinePredictor():
    def __init__(self, embedding_wrapper, input_mat, labels):

        self.glove_dim = 50
        self.embedding_wrapper = embedding_wrapper
        self.num_epochs = 10
        self.lr = 0.005
        self.inputs_placeholder = None
        self.labels_placeholder = None
        self.input_mat = input_mat
        self.labels = labels
        self.max_length = input_mat.shape[1]
        self.batch_size = 50
        self.num_hidden = 256
        self.num_layers = 3
        self.num_classes = 11
        self.loss = 0
        self.accuracy = .0
        self.train_len = input_mat.shape[0]


    def add_placeholders(self):
        self.inputs_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_length))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.num_classes))

    def create_feed_dict(self, inputs, labels):
        feed_dict = {
            self.inputs_placeholder : inputs,
            self.labels_placeholder : labels
        }
        return feed_dict

    def return_embeddings(self):
        data = np.load('trimmed_glove.6B.%dd.npz' % self.glove_dim)
        embeddings = tf.Variable(data['glove'])
        final_embeddings = tf.nn.embedding_lookup(embeddings, self.inputs_placeholder)
        final_embeddings = tf.reshape(final_embeddings, (-1, self.max_length, self.glove_dim)) 
        return final_embeddings

    def add_prediction_op(self, embeddings):
        multi_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.num_hidden) for _ in range(self.num_layers)])
        _, state = tf.nn.dynamic_rnn(multi_cell, embeddings, dtype=tf.float64)
        self.logits = layers.fully_connected(state[-1], self.num_classes, None)
        self.preds = tf.nn.softmax(self.logits)

    def add_loss_op(self):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels_placeholder)
        self.loss = tf.reduce_mean(loss)
    
    def add_optimization(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)

    def add_accuracy_op(self):
        discrete_preds = tf.argmax(self.preds, 1)
        discrete_labels = tf.argmax(self.labels_placeholder, 1)
        correct_preds = tf.equal(discrete_preds, discrete_labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

    def initialize_model(self):
        self.add_placeholders()
        embeddings = self.return_embeddings()
        logger.info("Running baseline...",)
        self.add_prediction_op(embeddings)
        self.add_loss_op()
        self.add_optimization()
        self.add_accuracy_op()

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs=inputs_batch, labels=labels_batch)
        loss, accuracy = sess.run([self.loss, self.accuracy], feed_dict=feed)
        _ = sess.run([self.train_op], feed_dict=feed)
        return loss, accuracy

    def run_epoch(self, sess):
        prog = Progbar(target=1 + int(self.train_len / self.batch_size))
        count = 0

        # TODO the loop below is a batch loop 
        batches = make_batches(self.batch_size, self.input_mat, self.labels)
        for batch in batches:
            tf.get_variable_scope().reuse_variables()
            loss, accuracy = self.train_on_batch(sess, batch[0], batch[1])
            prog.update(count + 1, [("train loss", loss), ("accuracy", accuracy)])
            count += 1
        return accuracy
    
    def fit(self, sess, saver):
        best_score = 0.
        epoch_scores = []
        for epoch in range(self.num_epochs):
            tf.get_variable_scope().reuse_variables()
            print("\nEpoch %d out of %d" % (epoch + 1, self.num_epochs))
            score = self.run_epoch(sess)
            if score > best_score:
                best_score = score
                if saver:
                    print("New best score! Saving model in %s" % self.model_output)
                    saver.save(sess, self.model_output)
            epoch_scores.append(score)



def build_model(embedding_wrapper, input_mat, labels):
    with tf.variable_scope('baseline_model'):
        logger.info("Building model...",)
        start = time.time()
        model = BaselinePredictor(embedding_wrapper, input_mat, labels)
        model.initialize_model()
        tf.get_variable_scope().reuse_variables()
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            model.fit(session, saver)        

def return_files(path):
    return [path+f for f in os.listdir(path) if (not f.startswith('missing_files') and not f.startswith('.'))]


def build_data(embedding_wrapper):
    dataset = []
    for file in return_files(train_datapath):
        idxs = []
        with open(file, 'r') as f:
            text = f.read()
            text = text.split()
            idxs = [embedding_wrapper.get_value(word) for word in text]
        dataset.append(idxs)


    arr = []
    with open(label_data, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            arr.append([1 if lang_dict[row[3]] == i else 0 for i in range(len(lang_dict))])
    arr = np.asarray(arr)
    with open('labels.dat', 'wb') as v:
        pickle.dump(arr, v)
        v.close()    

    res, lengths = pad_sequences(dataset)
    with open('padded_data.dat', 'wb') as v:
        pickle.dump(res, v)
        v.close()


def main():
    mydir = os.path.join(os.getcwd(), train_name)
    os.makedirs(mydir)

    embedding_wrapper = EmbeddingWrapper()
    embedding_wrapper.build_vocab()
    embedding_wrapper.process_glove()
    if not gfile.Exists(os.getcwd() + '/padded_data.dat'):
        print('build data')
        build_data(embedding_wrapper)
    try:
        input_mat = pickle.load(open('padded_data.dat', 'rb'))
        labels = pickle.load(open('labels.dat', 'rb'))
    except EOFError:
        print('No data')
        return {}
    build_model(embedding_wrapper, input_mat, labels)


if __name__ == "__main__":
    main()

