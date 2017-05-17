import os
import numpy as np
import heapq
import logging
import time
from scipy.signal import argrelextrema
from build_embedding import EmbeddingWrapper
import csv
import gzip
import cPickle as pickle

#from pointer_network import PointerCell

from utils import pad_sequences, make_batches
from progbar import Progbar

import tensorflow as tf
from tensorflow.python.platform import gfile

logger = logging.getLogger("hw3.q2")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

t = time.localtime()
timeString  = time.strftime("%Y%m%d%H%M%S", t)
train_name = "baseline_" + str(time.time())
logs_path = os.getcwd() + '/tf_log/'
train = False
train_datapath = os.getcwd() + '/data/speech_transcriptions/train/tokenized/'
label_data = os.getcwd() + '/data/labels/train/labels.train.csv'

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
        self.lr = 0.0001
        self.inputs_placeholder = None
        self.labels_placeholder = None
        self.input_mat = input_mat
        self.labels = labels
        self.max_length = input_mat.shape[1]
        self.batch_size = 50
        self.num_hidden = 5
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

        logits = None 


        cell = tf.contrib.rnn.LSTMCell(self.num_hidden)
        W = tf.get_variable(name = "W", shape = (self.num_hidden, self.num_classes), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
        b = tf.get_variable(name = "b", shape = (self.num_classes), initializer = tf.constant_initializer(0), dtype=tf.float64)

        # x = self.inputs_placeholder
        shape = tf.shape(embeddings)

        outputs, state = tf.nn.dynamic_rnn(cell, embeddings, dtype=tf.float64)
        outputs = tf.reshape(outputs, [-1, self.num_hidden])

        logits = tf.matmul(outputs, W) + b
        logits = tf.reshape(logits, [shape[0], shape[1], self.num_classes])
        self.preds = logits
        # logits = tf.reshape(logits, [shape[0], shape[1], self.num_classes])

    def add_loss_op(self):
        print self.labels_placeholder.get_shape()
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.preds, labels=self.labels_placeholder)
        self.loss = tf.reduce_sum(loss)
        # return self.loss
    
    def add_optimization(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)
        # return self.train_op       

    def add_accuracy_op(self):
        # number of max(logits) == labels / len(labels)

        discrete_preds = tf.argmax(self.preds, 1)
        discrete_labels = tf.argmax(self.labels_placeholder, 1)
        self.accuracy = tf.reduce_mean(tf.cast(discrete_preds == discrete_labels, dtype=tf.float32))
        # self.accuracy = tf.metrics.accuracy(self.labels_placeholder, discrete_preds)


    def initialize_model(self):
        self.add_placeholders()
        embeddings = self.return_embeddings()
        logger.info("Running baseline...",)
        self.add_prediction_op(embeddings)
        self.add_loss_op()
        self.add_optimization()
        self.add_accuracy_op()

        return self.preds, self.loss, self.train_op, self.add_accuracy_op  

    # def evaluate_epoch():
    #     # TODO  

    def train_on_batch(self, sess, inputs_batch, labels_batch):

        feed = self.create_feed_dict(inputs=inputs_batch, labels=labels_batch)

        _, loss, accuracy = sess.run([self.train_op, self.loss, self.accuracy], feed_dict=feed)
        return loss, accuracy


    def run_epoch(self, sess):
        prog = Progbar(target=1 + int(self.train_len / self.batch_size))
        count = 0

        # TODO the loop below is a batch loop 
        batches = make_batches(self.batch_size, self.input_mat, self.labels)
        for batch in batches:
            print batch[0].shape
            print batch[1].shape
            tf.get_variable_scope().reuse_variables()
            loss, accuracy = self.train_on_batch(sess, batch[0], batch[1])
            prog.update(count + 1, [("train loss", loss), ("accuracy", accuracy)])
            count += 1
        print("")

        print("Evaluating on development data")
        # exact_match, entity_scores = self.evaluate_epoch(sess)
        print("Entity level end_exact_match/start_exact_match/P/R/F1: %.2f/%.2f/%.2f/%.2f", exact_match[0], exact_match[1], entity_scores[0], entity_scores[1], entity_scores[2])


    
    def fit(self, sess, saver):
        best_score = 0.
        epoch_scores = []
        for epoch in range(self.num_epochs):
            tf.get_variable_scope().reuse_variables()
            print("Epoch %d out of %d" % (epoch + 1, self.num_epochs))
            score = self.run_epoch(sess)
            if score > best_score:
                best_score = score
                if saver:
                    print("New best score! Saving model in %s" % self.model_output)
                    saver.save(sess, self.model_output)
            epoch_scores.append(score)
            print("")



def build_model(embedding_wrapper, input_mat, labels):
    with tf.variable_scope('baseline_model'):
        logger.info("Building model...",)
        start = time.time()
        model = BaselinePredictor(embedding_wrapper, input_mat, labels)
        preds, loss, train_op, acc = model.initialize_model()
        init = tf.global_variables_initializer()
        tf.get_variable_scope().reuse_variables()
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(init)
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
    with open(label_data, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            arr.append([1 if lang_dict[row[3]] == i else 0 for i in xrange(len(lang_dict))])
    print arr
    arr = np.asarray(arr)
    print "after"
    print arr
    with open('labels.dat', 'w') as v:
        pickle.dump(arr, v)
        v.close()    



    res, lengths = pad_sequences(dataset)
    with open('padded_data.dat', 'w') as v:
        pickle.dump(res, v)
        v.close()


def main():
    mydir = os.path.join(os.getcwd(), train_name)
    os.makedirs(mydir)

    embedding_wrapper = EmbeddingWrapper()
    embedding_wrapper.build_vocab()
    embedding_wrapper.process_glove()
    if not gfile.Exists(os.getcwd() + '/padded_data.dat'):
        build_data(embedding_wrapper)
    input_mat = pickle.load(open('padded_data.dat', 'r'))
    labels = pickle.load(open('labels.dat', 'r'))
    build_model(embedding_wrapper, input_mat, labels)


if __name__ == "__main__":
    main()

