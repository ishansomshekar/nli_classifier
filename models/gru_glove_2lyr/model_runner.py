#Python imports
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#tf imports
import tensorflow as tf
from tensorflow.python.platform import gfile

#our imports
module_home = os.environ['NLI_PATH']
sys.path.insert(0, module_home)
from utils.data_utils import *

import model_config
from model_definition import BaselinePredictor

def train_model(train_data, dev_data):
    model = BaselinePredictor(train_data, dev_data)
    model.initialize_model()
    tf.get_variable_scope().reuse_variables()
    saver = tf.train.Saver()
    with tf.Session() as session:
        writer = tf.summary.FileWriter(model_config.graph_dir, session.graph)
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(model_config.continue_checkpoint + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
        session.run(tf.global_variables_initializer())
        last_step = model.global_step.eval()
        epoch_train_scores, epoch_dev_scores = model.fit(session, saver, writer, last_step)
        print("Best score for each epoch:")
        for epoch, score in enumerate(epoch_train_scores):
            print('Epoch', epoch, ' - Train score:', epoch_train_scores[epoch], ' - Dev score:', epoch_dev_scores[epoch])


def prep_data():
    ensure_dir('checkpoints')
    ensure_dir(model_config.best_checkpoint)
    ensure_dir(model_config.continue_checkpoint)
    ensure_dir(model_config.processed_data_path)

    embedding_wrapper = model_config.get_embedding_wrapper()
    embedding_wrapper.build_vocab(model_config.vocab_path)
    embedding_wrapper.process_embeddings(model_config.embeddings_path)
    if not gfile.Exists(model_config.train_paths['inputs_out']) or not gfile.Exists(model_config.dev_paths['inputs_out']):
        print('build data')
        build_data_partition(model_config.train_paths, [embedding_wrapper])
        build_data_partition(model_config.dev_paths, [embedding_wrapper])

    train_data = load_data(model_config.train_paths)
    dev_data = load_data(model_config.dev_paths)

    return train_data, dev_data


def main():
    print("Running model, call with --fresh to clear preprocessed data from previous runs")
    if len(sys.argv) > 1 and sys.argv[1] == "--fresh":
        print("Run with --fresh: clearing previously processed data")
        clear_data(model_config.processed_data_path)
    train_data, dev_data = prep_data()
    with tf.variable_scope('baseline_model'):
        train_model(train_data, dev_data)


if __name__ == "__main__":
    main()
