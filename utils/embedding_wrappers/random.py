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

module_home = os.environ['NLI_PATH']
sys.path.insert(0, module_home)

from utils.data_utils import return_files


class RandomEmbeddingWrapper(object):
    def __init__(self, dim=50):
        self.name = 'word'
        self.vocab = None
        self.reverse_vocab = None
        self.embeddings = None
        self.embedding_dim = dim
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
                    if file_count % 100 == 0:
                        print ("finished reading %d transcripts" % file_count)

            vocab[self.unk] = idx
            reverse_vocab += [self.unk]
            idx += 1
            vocab[self.pad] = idx
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
        """
        :param vocab_list: [vocab]
        :return:
        """
        if not gfile.Exists(embeddings_path):
            print("build glove")
            random_embeddings = np.random.rand(len(self.vocab), self.embedding_dim)
            np.savez_compressed(embeddings_path, embedding=random_embeddings)
            print("saved random embeddings matrix at: {}".format(embeddings_path))
            self.embeddings = random_embeddings

    def get_value(self, word):
        if word in self.vocab:
            return self.vocab[word]
        else:
            return self.vocab[self.unk]

    def get_indices(self, text):
        words = text.split()
        indices = [self.get_value(word) for word in text.split()]
        return indices
