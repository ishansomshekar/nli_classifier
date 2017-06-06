import logging
import os
module_home = os.environ['NLI_PATH']

from utils.embedding_wrappers.random import RandomEmbeddingWrapper
from utils.embedding_wrappers.glove import GloveEmbeddingWrapper
from utils.embedding_wrappers.one_hot import OneHotEmbeddingWrapper
from utils.embedding_wrappers.char_level import CharLevelEmbeddingWrapper
from utils.embedding_wrappers.pos import PosEmbeddingWrapper

from utils.data_utils import get_script_path

# data

processed_data_path = os.path.join(get_script_path(), 'processed/')
best_checkpoint = os.path.join(module_home, 'checkpoints/word_pos_hybrid_best')
continue_checkpoint = os.path.join(module_home, 'checkpoints/word_pos_hybrid')

logs_path = os.path.join(processed_data_path, './tf_log')

word_vocab_path = os.path.join(processed_data_path, 'vocab.dat')
pos_vocab_path = os.path.join(processed_data_path, 'pos_vocab.dat')
word_embeddings_path = os.path.join(processed_data_path, 'embeddings.npz')
pos_embeddings_path = os.path.join(processed_data_path, 'pos_embeddings.npz')



train_paths = {
        'inputs_in': os.path.join(module_home, 'data/speech_transcriptions/train/tokenized/'),
        'labels_in': os.path.join(module_home, 'data/labels/train/labels.train.csv'),
        'inputs_out': [os.path.join(processed_data_path, 'train_word_inputs.dat'),
                       os.path.join(processed_data_path, 'train_pos_inputs.dat')],
        'seq_lens_out': os.path.join(processed_data_path, 'train_seq_lens.dat'),
        'labels_out': os.path.join(processed_data_path, 'train_labels.dat'),
        'max_len_out': os.path.join(processed_data_path, 'train_max_len.dat')
        }

dev_paths = {
        'inputs_in': os.path.join(module_home, 'data/speech_transcriptions/dev/tokenized/'),
        'labels_in': os.path.join(module_home, 'data/labels/dev/labels.dev.csv'),
        'inputs_out': [os.path.join(processed_data_path, 'dev_word_inputs.dat'),
                       os.path.join(processed_data_path, 'dev_pos_inputs.dat')],
        'seq_lens_out': os.path.join(processed_data_path, 'dev_seq_lens.dat'),
        'labels_out': os.path.join(processed_data_path, 'dev_labels.dat'),
        'max_len_out': os.path.join(processed_data_path, 'dev_max_len.dat')
        }

num_classes = 11
log_frequency = 100

# model hyperparameters
num_epochs = 200
batch_size = 64

num_hidden = 256
num_layers = 1
word_embedding_dim = 100

learning_rate = 2e-4
l2_rate = 4e-3
dropout_keep_prob = 0.5

#

multi_input = True

def get_embedding_wrappers():
    #return CharLevelEmbeddingWrapper() # for char level NN
    #return GloveEmbeddingWrapper() # for GloVe vector embeddings
    #return OneHotEmbeddingWrapper() # for one-hot word embeddings (warning: slow)
    #return RandomEmbeddingWrapper() # for random initialized word embeddings
    return [RandomEmbeddingWrapper(word_embedding_dim), PosEmbeddingWrapper()]

def get_embedding_paths():
    return [word_embeddings_path, pos_embeddings_path]

def get_vocab_paths():
    return [word_vocab_path, pos_vocab_path]

embeddings_trainable = False

def get_logger():
    logger = logging.getLogger("log_baseline")
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    return logger
