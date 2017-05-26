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
from model_definition import WordPOSPredictor

def train_model(train_data, dev_data):
    with tf.Session() as session:
        model = WordPOSPredictor(train_data, dev_data)
        model.initialize_model(session)
        tf.get_variable_scope().reuse_variables()
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('./graphs/word_pos', session.graph)
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(model_config.continue_checkpoint + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
        session.run(tf.global_variables_initializer())
        last_step = model.global_step.eval()
        epoch_train_scores, epoch_dev_scores = model.fit(session, saver, writer, last_step)
        print("Best score for each epoch:")
        for epoch, score in enumerate(epoch_train_scores):
            print('Epoch', epoch, ' - Train score:', epoch_train_scores[epoch], ' - Dev score:', epoch_dev_scores[epoch])


def filesNeeded():
    train_paths = model_config.train_paths['inputs_out']
    dev_paths = model_config.dev_paths['inputs_out']
    if model_config.multi_input:
        return not all(gfile.Exists(p) for p in train_paths + dev_paths)
    else:
        return not (gfile.Exists(train_paths) and gfile.Exists(dev_paths))


def prep_data():
    ensure_dir(model_config.best_checkpoint)
    ensure_dir(model_config.continue_checkpoint)
    ensure_dir(model_config.processed_data_path)

    embedding_wrappers = model_config.get_embedding_wrappers()
    embedding_paths = model_config.get_embedding_paths()
    vocab_paths = model_config.get_vocab_paths()
    for ew, em_path, voc_path in zip(embedding_wrappers, embedding_paths, vocab_paths):
        ew.build_vocab(voc_path)
        ew.process_embeddings(em_path)
    if filesNeeded():
        print('build data called')
        build_data_partition(model_config.train_paths, embedding_wrappers)
        build_data_partition(model_config.dev_paths, embedding_wrappers)

    train_data = load_data(model_config.train_paths, model_config.multi_input)
    dev_data = load_data(model_config.dev_paths, model_config.multi_input)
    print("Train data inputs0 max: %d" % np.max(train_data['inputs'][:,0,:]))
    print("Train data inputs1 max: %d" % np.max(train_data['inputs'][:,1,:]))

    return train_data, dev_data


def main():
    print "Running model, call with --fresh to clear preprocessed data from previous runs"
    if len(sys.argv) > 1 and sys.argv[1] == "--fresh":
        print "Run with --fresh: clearing previously processed data"
        clear_data(model_config.processed_data_path)
    print("Prepping data")
    train_data, dev_data = prep_data()
    print("Prep done")
    with tf.variable_scope('baseline_model'):
        train_model(train_data, dev_data)


if __name__ == "__main__":
    main()
