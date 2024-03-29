import logging
import os
# module_home = os.environ['NLI_PATH']
module_home = '/Users/Chip/dev/cs224s/nli_classifier'

from utils.embedding_wrappers.random import RandomEmbeddingWrapper
from utils.embedding_wrappers.glove import GloveEmbeddingWrapper
from utils.embedding_wrappers.one_hot import OneHotEmbeddingWrapper
from utils.embedding_wrappers.char_level import CharLevelEmbeddingWrapper

from utils.data_utils import get_script_path

# data
processed_data_path = os.path.join(get_script_path(), 'processed/')

best_checkpoint = os.path.join(processed_data_path, 'checkpoints/best')
continue_checkpoint = os.path.join(processed_data_path, 'checkpoints/run')
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
log_frequency = 1000

# model hyperparameters

num_epochs = 50
batch_size = 64

num_hidden = 256
num_layers = 2

learning_rate = 1e-3
l2_rate = 1e-4
dropout_keep_prob = 0.5

#

def get_embedding_wrapper():
    return CharLevelEmbeddingWrapper() # for char level NN
    #return GloveEmbeddingWrapper() # for GloVe vector embeddings
    #return OneHotEmbeddingWrapper() # for one-hot word embeddings (warning: slow)
    #return RandomEmbeddingWrapper() # for random initialized word embeddings

embeddings_trainable = False

def get_logger():
    logger = logging.getLogger("log_baseline")
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    return logger
