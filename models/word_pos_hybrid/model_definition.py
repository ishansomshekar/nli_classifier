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
import model_config


class WordPOSPredictor():
    def __init__(self, train_data, dev_data):

        self.logger = model_config.get_logger()

        self.word_inputs_placeholder = None
        self.pos_inputs_placeholder = None
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
        self.word_inputs_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        self.pos_inputs_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        self.seq_lens_placeholder = tf.placeholder(tf.int32, shape=(None))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.num_classes))
	self.dropout_keep_prob_placeholder = tf.placeholder(tf.float64)


    def create_feed_dict(self, inputs, lens, labels, keep_prob):
        print inputs
        feed_dict = {
            self.pos_inputs_placeholder : inputs['pos'],
            self.word_inputs_placeholder : inputs['word'],
            self.seq_lens_placeholder: lens,
            self.labels_placeholder : labels,
	    self.dropout_keep_prob_placeholder : keep_prob
        }
        return feed_dict


    def add_embeddings(self, session):
        word_embedding_data = load_embedding_data(model_config.word_embeddings_path)
        pos_embedding_data = load_embedding_data(model_config.pos_embeddings_path)

        word_embeddings = tf.Variable(tf.constant(0.0, shape=word_embedding_data.shape), name="word_embeddings", trainable=model_config.embeddings_trainable)
        word_embedding_placeholder = tf.placeholder(tf.float32, word_embedding_data.shape)

        final_word_embeddings = tf.nn.embedding_lookup(word_embeddings, self.word_inputs_placeholder)

        word_embedding_dim = word_embedding_data.shape[1]
        word_max_seq_len = tf.shape(self.word_inputs_placeholder)[0]

        word_embedding_init = word_embeddings.assign(word_embedding_placeholder)

        word_embeddings = tf.cast(tf.reshape(word_embeddings, (-1, word_max_seq_len, word_embedding_dim)), tf.float64)



        pos_embeddings = tf.Variable(tf.constant(0.0, shape=pos_embedding_data.shape), name="pos_embeddings", trainable=False)
        pos_embedding_placeholder = tf.placeholder(tf.float32, pos_embedding_data.shape)

        final_pos_embeddings = tf.nn.embedding_lookup(pos_embeddings, self.pos_inputs_placeholder)

        pos_embedding_dim = pos_embedding_data.shape[1]
        pos_max_seq_len = tf.shape(self.pos_inputs_placeholder)[0]

        pos_embedding_init = pos_embeddings.assign(pos_embedding_placeholder)

        pos_embeddings = tf.cast(tf.reshape(pos_embeddings, (-1, pos_max_seq_len, pos_embedding_dim)), tf.float64)

        self.full_embeddings = tf.concat([word_embeddings, pos_embeddings], axis=2)

        session.run([word_embedding_init, pos_embedding_init], feed_dict={word_embedding_placeholder:word_embedding_data, pos_embedding_placeholder:pos_embedding_data})


    def add_prediction_op(self):
	gru_cell = tf.contrib.rnn.MultiRNNCell([
            tf.contrib.rnn.DropoutWrapper(
		tf.contrib.rnn.GRUCell(self.num_hidden),
		self.dropout_keep_prob_placeholder)
            for _ in xrange(self.num_layers)])

        _, state = tf.nn.dynamic_rnn(
                gru_cell,
                self.full_embeddings,
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

    def initialize_model(self, session):
        self.add_placeholders()
        self.add_embeddings(session)
        self.logger.info("Running POS-Word hybrid model...",)
        self.add_prediction_op()
        self.add_loss_op()
        self.add_optimization()
        self.add_accuracy_op()
        self.add_summaries()

    def train_on_batch(self, sess, inputs_batch, lens_batch, labels_batch):
        feed = self.create_feed_dict(inputs=inputs_batch, lens=lens_batch, labels=labels_batch, keep_prob=model_config.dropout_keep_prob)
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
        batch = make_batches(self.dev_len, self.dev_data)[0]
	tf.get_variable_scope().reuse_variables()
        feed = self.create_feed_dict(inputs=batch[0], lens=batch[1], labels=batch[2], keep_prob=1.0)
        loss, accuracy = sess.run([self.loss, self.accuracy], feed_dict=feed)
        if accuracy > best_score:
            best_score = accuracy
            print("\nNew best score! Saving model in %s" % config.best_checkpoint)
            saver.save(sess, config.best_checkpoint + '/baseline_lstm')
        return accuracy, best_score

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
