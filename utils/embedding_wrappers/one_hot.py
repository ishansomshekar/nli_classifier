from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pickle

from tensorflow.python.platform import gfile
from tqdm import *
import numpy as np
from os.path import join as pjoin

# module_home = os.environ['NLI_PATH']
module_home = '/Users/Chip/dev/cs224s/nli_classifier'
sys.path.insert(0, module_home)


def return_files(path):
    return [path+f for f in os.listdir(path) if (not f.startswith('missing_files') and not f.startswith('.'))]

def return_dir(path):
    return [path+f for f in os.listdir(path) if (not f.startswith('.'))]

class OneHotEmbeddingWrapper(object):
    def __init__(self):
        self.vocab = None
        self.reverse_vocab = None
        self.embeddings = None
        self.num_tokens = 0
        self.unk = "UNK"
        self.pad = "<PAD>"
        self.data_path = pjoin(module_home, 'data/speech_transcriptions/train/tokenized/')


    def build_vocab(self, path):
        if not gfile.Exists(path):
            print ("building vocabulary for all files")
            dataset_len = 0
            vocab = dict()
            reverse_vocab = []
            idx = 0
            wordcounter = 0
            file_count = 0
            for file in return_files(self.data_path):
                with open(file, 'r') as f:
                    for i, line in enumerate(f):
                        words = line.split()
                        for word in words:
                            wordcounter += 1
                            if not word in vocab:
                                vocab[word] = idx
                                reverse_vocab +=[word]
                                idx += 1
                    file_count+=1
                    if file_count % 500 == 0:
                        print ("finished reading %d transcripts" % file_count)

            vocab[self.unk] = idx
            vocab[self.pad] = idx+1
            reverse_vocab += [self.unk]
            reverse_vocab += [self.pad]
            wordcounter += 2

            self.vocab = vocab
            self.reverse_vocab = reverse_vocab
            self.num_tokens = wordcounter
            print( "finished building vocabulary of size %d for all files" %wordcounter)
        else:
            self.vocab = pickle.load(open(path, 'r'))
            self.reverse_vocab = None
            self.num_tokens = len(self.vocab)


    def process_embeddings(self, embeddings_path):
        if not gfile.Exists(embeddings_path):
            print("build one hot vectors")
            embeddings = np.diag(np.ones(len(self.vocab)))
            np.savez_compressed(embeddings_path, embedding=embeddings)

    def get_value(self, word):
        if word in self.vocab:
            return self.vocab[word]
        else:
            return self.vocab[self.unk]

    def get_indices(self, text):
        return [self.get_value(word) for word in text.split()]
