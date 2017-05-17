#Python imports
import os
import numpy as np
import time
import csv
import cPickle as pickle
import logging
import csv
import logging
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#tf imports
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.platform import gfile

#our imports
module_home = os.environ['NLI_PATH']
sys.path.insert(0, module_home)

from utils.progbar import Progbar
from utils.embedding_wrapper import EmbeddingWrapper
from utils.data_utils import *
import model_config


class BaselinePredictor():
    def __init__(self, embedding_wrapper, input_mat, lens, labels):
        self.embedding_wrapper = embedding_wrapper
        self.logger = model_config.get_logger()

        self.inputs_placeholder = None
        self.seq_lens_placeholder = None
        self.labels_placeholder = None

        self.input_mat = input_mat
        self.lens = lens
        self.labels = labels

        self.max_length = input_mat.shape[1]
        self.train_len = input_mat.shape[0]

        self.num_classes = model_config.num_classes

        self.glove_dim = model_config.glove_dim
        self.num_epochs = model_config.num_epochs
        self.batch_size = model_config.batch_size

        self.lr = model_config.learning_rate
        self.l2r = model_config.l2_rate

        self.num_hidden = model_config.num_hidden
        self.num_layers = model_config.num_layers

        self.loss = 0
        self.accuracy = .0
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def add_placeholders(self):
        self.inputs_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_length))
        self.seq_lens_placeholder = tf.placeholder(tf.int32, shape=(None))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.num_classes))

    def create_feed_dict(self, inputs, lens, labels):
        feed_dict = {
            self.inputs_placeholder : inputs,
            self.seq_lens_placeholder: lens,
            self.labels_placeholder : labels
        }
        return feed_dict

    def return_embeddings(self):
        glove_data = load_glove_data(model_config.processed_data_path, self.glove_dim)
        embeddings = tf.Variable(glove_data['glove'])
        final_embeddings = tf.nn.embedding_lookup(embeddings, self.inputs_placeholder)
        final_embeddings = tf.reshape(final_embeddings, (-1, self.max_length, self.glove_dim))
        return final_embeddings

    def add_prediction_op(self, embeddings):
        gru_cell = tf.contrib.rnn.GRUCell(self.num_hidden)
        multi_cell = tf.contrib.rnn.MultiRNNCell([gru_cell] * self.num_layers)
        _, state = tf.nn.dynamic_rnn(multi_cell, embeddings, sequence_length=self.seq_lens_placeholder, dtype=tf.float64)
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
        self.logger.info("Running baseline...",)
        self.add_prediction_op(embeddings)
        self.add_loss_op()
        self.add_optimization()
        self.add_accuracy_op()
        self.add_summaries()

    def train_on_batch(self, sess, inputs_batch, lens_batch, labels_batch):
        feed = self.create_feed_dict(inputs=inputs_batch, lens=lens_batch, labels=labels_batch)
        _, loss, accuracy, summary = sess.run([self.train_op, self.loss, self.accuracy, self.summary_op], feed_dict=feed)
        return loss, accuracy, summary

    def run_epoch(self, sess, saver, best_score, writer, last_step):
        prog = Progbar(target=1 + int(self.train_len / self.batch_size))
        count = 0

        batches = make_batches(self.batch_size, self.input_mat, self.lens, self.labels)
        for index, batch in enumerate(batches):
            tf.get_variable_scope().reuse_variables()
            loss, accuracy, summary = self.train_on_batch(sess, batch[0], batch[1], batch[2])
            writer.add_summary(summary, global_step=index + last_step)
            prog.update(count + 1, [("train loss", loss), ("accuracy", accuracy)])
            count += 1
            if accuracy > best_score:
                best_score = accuracy
                if saver:
                    print("\nNew best score! Saving model in %s" % model_config.best_checkpoint)
                    saver.save(sess, model_config.best_checkpoint + '/baseline_lstm', index + last_step)
            if (index + 1) % model_config.log_frequency == 0:
                saver.save(sess, model_config.continue_checkpoint + '/baseline_lstm', index + last_step)
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


def build_model(embedding_wrapper, input_mat, lens, labels):
    with tf.variable_scope('baseline_model'):
        model = BaselinePredictor(embedding_wrapper, input_mat, lens, labels)
        model.initialize_model()
        tf.get_variable_scope().reuse_variables()
        saver = tf.train.Saver()
        with tf.Session() as session:
            writer = tf.summary.FileWriter('./graphs/baseline_lstm', session.graph)
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(model_config.continue_checkpoint + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
            session.run(tf.global_variables_initializer())
            last_step = model.global_step.eval()
            epoch_scores = model.fit(session, saver, writer, last_step)

    print("Best score for each epoch:")
    for epoch, score in enumerate(epoch_scores):
        print(epoch, ':', score)

def main():
    ensure_dir('checkpoints')
    ensure_dir(model_config.best_checkpoint)
    ensure_dir(model_config.continue_checkpoint)

    embedding_wrapper = EmbeddingWrapper()
    embedding_wrapper.build_vocab()
    embedding_wrapper.process_glove()


    if not gfile.Exists(model_config.padded_data_path):
        print('build data')
        build_data(model_config.train_raw_path, model_config.labels_raw_path, model_config.processed_data_path, embedding_wrapper)
    try:
        input_mat = pickle.load(open(model_config.padded_data_path, 'rb'))
        lens = pickle.load(open(model_config.seq_lens_data_path, 'rb'))
        labels = pickle.load(open(model_config.labels_data_path, 'rb'))
    except EOFError:
        print('No data')
        return {}
    build_model(embedding_wrapper, input_mat, lens, labels)


if __name__ == "__main__":
    main()

