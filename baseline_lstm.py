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
from progbar import Progbar
import utils

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
        self.batch_size = 64
        self.num_hidden = 128
        self.num_layers = 1
        self.num_classes = 11
        self.loss = 0
        self.accuracy = .0
        self.train_len = input_mat.shape[0]
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

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
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def add_accuracy_op(self):
        discrete_preds = tf.argmax(self.preds, 1)
        discrete_labels = tf.argmax(self.labels_placeholder, 1)
        correct_preds = tf.equal(discrete_preds, discrete_labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

    def add_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)
            self.summary_op = tf.summary.merge_all()

    def initialize_model(self):
        self.add_placeholders()
        embeddings = self.return_embeddings()
        logger.info("Running baseline...",)
        self.add_prediction_op(embeddings)
        self.add_loss_op()
        self.add_optimization()
        self.add_accuracy_op()
        self.add_summaries()

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs=inputs_batch, labels=labels_batch)
        _, loss, accuracy, summary = sess.run([self.train_op, self.loss, self.accuracy, self.summary_op], feed_dict=feed)
        return loss, accuracy, summary

    def run_epoch(self, sess, saver, best_score, writer, last_step):
        prog = Progbar(target=1 + int(self.train_len / self.batch_size))
        count = 0

        # TODO the loop below is a batch loop 
        batches = utils.make_batches(self.batch_size, self.input_mat, self.labels)
        for index, batch in enumerate(batches):
            tf.get_variable_scope().reuse_variables()
            loss, accuracy, summary = self.train_on_batch(sess, batch[0], batch[1])
            writer.add_summary(summary, global_step=index + last_step)
            prog.update(count + 1, [("train loss", loss), ("accuracy", accuracy)])
            count += 1
            if accuracy > best_score:
                best_score = accuracy
                if saver:
                    print("\nNew best score! Saving model in %s" % config.best_checkpoint)
                    saver.save(sess, config.best_checkpoint + '/baseline_lstm', index + last_step)
            if (index + 1) % config.log_frequency == 0:
                saver.save(sess, config.continue_checkpoint + '/baseline_lstm', index + last_step)
        return accuracy, best_score
    
    def fit(self, sess, saver, writer, last_step):
        best_score = 0.
        epoch_scores = []
        for epoch in range(self.num_epochs):
            tf.get_variable_scope().reuse_variables()
            print("\nEpoch %d out of %d" % (epoch + 1, self.num_epochs))
            accuracy, best_score = self.run_epoch(sess, saver, best_score, writer, last_step)
            epoch_scores.append(best_score)
        return epoch_scores 

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

    res, lengths = utils.pad_sequences(dataset)
    with open('padded_data.dat', 'wb') as v:
        pickle.dump(res, v)
        v.close()

def build_model(embedding_wrapper, input_mat, labels):
    with tf.variable_scope('baseline_model'):
        logger.info("Building model...",)
        model = BaselinePredictor(embedding_wrapper, input_mat, labels)
        model.initialize_model()
        tf.get_variable_scope().reuse_variables()
        saver = tf.train.Saver()
        with tf.Session() as session:
            writer = tf.summary.FileWriter('./graphs/baseline_lstm', session.graph)
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.continue_checkpoint + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
            session.run(tf.global_variables_initializer())
            last_step = model.global_step.eval()
            epoch_scores = model.fit(session, saver, writer, last_step)

    print("Best score for each epoch:")
    for epoch, score in enumerate(epoch_scores):
        print(epoch, ':', score)

def main():
    mydir = os.path.join(os.getcwd(), train_name)
    os.makedirs(mydir)
    utils.make_dir('checkpoints')
    utils.make_dir(config.best_checkpoint)
    utils.make_dir(config.continue_checkpoint)

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

