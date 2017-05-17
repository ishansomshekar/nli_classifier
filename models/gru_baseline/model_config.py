import logging
import os
module_home = os.environ['NLI_PATH']

# data
best_checkpoint = 'checkpoints/best'
continue_checkpoint = 'checkpoints/run1'

train_raw_path = os.path.join(module_home, 'data/speech_transcriptions/train/tokenized/')
labels_raw_path = os.path.join(module_home, 'data/labels/train/labels.train.csv')

logs_path = os.path.join('./tf_log')

processed_data_path = os.path.join(module_home, 'data/processed/')

padded_data_path = os.path.join(processed_data_path, 'padded_data.dat')
seq_lens_data_path = os.path.join(processed_data_path, 'lens.dat')
labels_data_path = os.path.join(processed_data_path, 'labels.dat')

num_classes = 11
log_frequency = 100

# model hyperparameters
glove_dim = 50

num_epochs = 10
batch_size = 64

num_hidden = 128
num_layers = 1

learning_rate = 1e-3
l2_rate = 1e-4

#

def get_logger():
    logger = logging.getLogger("log_baseline")
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    return logger
