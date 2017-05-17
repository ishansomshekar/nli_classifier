import logging
import os
module_home = os.environ['NLI_PATH']

# data
best_checkpoint = 'checkpoints/best'
continue_checkpoint = 'checkpoints/run1'

logs_path = os.path.join('./tf_log')

processed_data_path = os.path.join(module_home, 'data/processed/')

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
