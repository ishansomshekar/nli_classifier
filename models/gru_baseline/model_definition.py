#Python imports
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#tf imports
import tensorflow as tf
import tensorflow.contrib.layers as layers

#our imports
module_home = os.environ['NLI_PATH']
sys.path.insert(0, module_home)

from utils.progbar import Progbar
from utils.data_utils import *
import models.gru_baseline.model_config as model_config


class BaselinePredictor():
    def __init__(self, train_data, dev_data):

        self.logger = model_config.get_logger()

        self.inputs_placeholder = None
        self.seq_lens_placeholder = None
        self.labels_placeholder = None

        self.train_data = train_data
        self.dev_data = dev_data

        self.max_length = max(train_data['max_len'], dev_data['max_len'])

        self.train_len = self.train_data['inputs'].shape[0]
        self.dev_len = self.dev_data['inputs'].shape[0]

        self.num_classes = model_config.num_classes

        self.num_epochs = model_config.num_epochs
        self.batch_size = model_config.batch_size

        self.lr = model_config.learning_rate
        self.l2r = model_config.l2_rate

        self.num_hidden = model_config.num_hidden
        self.num_layers = model_config.num_layers

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
        embedding_data = load_embedding_data(model_config.processed_data_path)
        self.embedding_dim = embedding_data.shape[1]
        print "__________________________________"
        print embedding_data.shape
        embeddings = tf.Variable(embedding_data)
        final_embeddings = tf.nn.embedding_lookup(embeddings, self.inputs_placeholder)
        final_embeddings = tf.cast(tf.reshape(final_embeddings, (-1, self.max_length, self.embedding_dim)), tf.float64)
        return final_embeddings


    def add_prediction_op(self, embeddings):
        gru_cell = tf.contrib.rnn.GRUCell(self.num_hidden)
        multi_cell = tf.contrib.rnn.MultiRNNCell([gru_cell] * self.num_layers)
        _, state = tf.nn.dynamic_rnn(
                multi_cell,
                embeddings,
                sequence_length=self.seq_lens_placeholder,
                dtype=tf.float64)

        self.logits = layers.fully_connected(
                inputs = state[-1],
                num_outputs = self.num_classes,
                activation_fn = None,
                weights_initializer = tf.contrib.layers.xavier_initializer(),
                weights_regularizer = tf.contrib.layers.l2_regularizer(self.l2r),
                biases_initializer = tf.zeros_initializer(),
                trainable = True)
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

        batches = make_batches(self.batch_size, self.train_data)
        for index, batch in enumerate(batches):
            tf.get_variable_scope().reuse_variables()
            loss, accuracy, summary = self.train_on_batch(sess, batch[0], batch[1], batch[2])
            writer.add_summary(summary, global_step=index + last_step)
            prog.update(count + 1, [("train loss", loss), ("accuracy", accuracy)])
            count += 1
            if accuracy > best_score:
                best_score = accuracy
            if (index + last_step + 1) % model_config.log_frequency == 0:
                saver.save(sess, model_config.continue_checkpoint + '/baseline_lstm', index + last_step)
        return accuracy, best_score

    def eval_dev(self, sess, saver, best_score):
        prog = Progbar(target=int(self.dev_len))
        count = 0
        total_accuracy = 0.0
        batches = make_batches(1, self.dev_data)
        for batch in batches:
            tf.get_variable_scope().reuse_variables()
            feed = self.create_feed_dict(inputs=batch[0], lens=batch[1], labels=batch[2])
            loss, accuracy = sess.run([self.loss, self.accuracy], feed_dict=feed)
            prog.update(count + 1, [("dev loss", loss), ("dev accuracy", accuracy)])
            total_accuracy += accuracy
            count += 1
        final_accuracy = total_accuracy / count
        if final_accuracy > best_score:
            best_score = final_accuracy
            print("\nNew best score! Saving model in %s" % config.best_checkpoint)
            saver.save(sess, config.best_checkpoint + '/baseline_lstm')
        return final_accuracy, best_score

    def fit(self, sess, saver, writer, last_step):
        best_dev_score = 0.0
        best_train_score = 0.0
        epoch_dev_scores = []
        epoch_train_scores = []
        for epoch in range(self.num_epochs):
            tf.get_variable_scope().reuse_variables()
            print("\nEpoch %d out of %d" % (epoch + 1, self.num_epochs))
            accuracy, best_train_score = self.run_epoch(sess, saver, best_train_score, writer, last_step)
            score, best_dev_score = self.eval_dev(sess, saver, best_dev_score)
            epoch_dev_scores.append(score)
            epoch_train_scores.append(best_train_score)
        return epoch_train_scores, epoch_dev_scores
