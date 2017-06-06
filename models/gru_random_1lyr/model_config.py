import logging
import os
module_home = os.environ['NLI_PATH']

from utils.embedding_wrappers.random import RandomEmbeddingWrapper
from utils.embedding_wrappers.glove import GloveEmbeddingWrapper
from utils.embedding_wrappers.one_hot import OneHotEmbeddingWrapper
from utils.embedding_wrappers.char_level import CharLevelEmbeddingWrapper

from utils.data_utils import get_script_path

# data
processed_data_path = os.path.join(get_script_path(), 'processed/')

# best_checkpoint = os.path.join(processed_data_path, 'checkpoints/best')
# continue_checkpoint = os.path.join(processed_data_path, 'checkpoints/run')

logs_path = os.path.join(processed_data_path, './tf_log')

vocab_path = os.path.join(processed_data_path, 'vocab.dat')
embeddings_path = os.path.join(processed_data_path, 'embeddings.npz')

train_paths = {
        'inputs_in': os.path.join(module_home, 'data/speech_transcriptions/train/tokenized/'),
        'labels_in': os.path.join(module_home, 'data/labels/train/labels.train.csv'),
        'inputs_out': os.path.join(processed_data_path, 'train_padded_inputs.dat'),
        'seq_lens_out': os.path.join(processed_data_path, 'train_seq_lens.dat'),
        'labels_out': os.path.join(processed_data_path, 'train_labels.dat'),
        'max_len_out': os.path.join(processed_data_path, 'train_max_len.dat')
        }

dev_paths = {
        'inputs_in': os.path.join(module_home, 'data/speech_transcriptions/dev/tokenized/'),
        'labels_in': os.path.join(module_home, 'data/labels/dev/labels.dev.csv'),
        'inputs_out': os.path.join(processed_data_path, 'dev_added_inputs.dat'),
        'seq_lens_out': os.path.join(processed_data_path, 'dev_seq_lens.dat'),
        'labels_out': os.path.join(processed_data_path, 'dev_labels.dat'),
        'max_len_out': os.path.join(processed_data_path, 'dev_max_len.dat')
        }

num_classes = 11
log_frequency = 100

# model hyperparameters

num_epochs = 10
batch_size = 64

num_hidden = 512
num_layers = 1

# learning_rate = 1e-4
# l2_rate = 1e-4
# dropout_keep_prob = 0.5

learning_rate = 2e-5
l2_rate = 1e-4
dropout_keep_prob = 0.6

gru_random_1lyr_
model_all = 'gru_random_1lyr_' + str(learning_rate) + '_l2_' + str(l2_rate) + '_drop_' + str(dropout_keep_prob)
best_checkpoint = os.path.join(module_home, 'checkpoints/' + model_all + '_best')
continue_checkpoint = os.path.join(module_home, 'checkpoints/' + model_all)
graph_dir = os.path.join(module_home, 'graphs/' + model_all)

# best_checkpoint = os.path.join(module_home, 'checkpoints/gru_random_1lyr_' + str(learning_rate) + '_l2_' + str(l2_rate) + '_drop_' + str(dropout_keep_prob) + '_best')
# continue_checkpoint = os.path.join(module_home, 'checkpoints/gru_random_1lyr_' + str(learning_rate) + '_l2_' + str(l2_rate) + '_drop_' + str(dropout_keep_prob))

def get_embedding_wrapper():
    #return CharLevelEmbeddingWrapper() # for char level NN
    #return GloveEmbeddingWrapper() # for GloVe vector embeddings
    #return OneHotEmbeddingWrapper() # for one-hot word embeddings (warning: slow)
    return RandomEmbeddingWrapper(300) # for random initialized word embeddings

embeddings_trainable = True

def get_logger():
    logger = logging.getLogger("log_baseline")
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    return logger
