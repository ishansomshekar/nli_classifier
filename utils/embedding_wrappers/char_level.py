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

from utils.data_utils import return_files


class CharLevelEmbeddingWrapper(object):
    def __init__(self):
        self.alphabet = None
        self.reverse_alphabet = None
        self.embeddings = None
        self.embedding_dim = 50
        self.num_chars = 0
        self.unk = "UNK"
        self.pad = "<PAD>"
        self.data_path = pjoin(module_home, 'data/speech_transcriptions/train/tokenized/')


    def build_vocab(self, path):
        if not gfile.Exists(path):
            print ("building alphabet for all files")
            dataset_len = 0
            alphabet = dict()
            reverse_alphabet = []
            idx = 0
            charcounter = 0
            file_count = 0
            for file in return_files(self.data_path):
                with open(file, 'r') as f:
                    for i, line in enumerate(f):
                        for char in line:
                            charcounter += 1
                            if char not in alphabet:
                                alphabet[char] = idx
                                reverse_alphabet += [char]
                                idx += 1
                    file_count+=1
                    if file_count % 100 == 0:
                        print ("finished reading %d transcripts" % file_count)

            alphabet[self.unk] = idx
            reverse_alphabet += [self.unk]
            idx += 1
            alphabet[self.pad] = idx
            reverse_alphabet += [self.pad]
            charcounter += 2

            self.alphabet = alphabet
            self.reverse_alphabet = reverse_alphabet
            self.num_chars = charcounter
            print( "finished building alphabet of size %d for all files" % charcounter)
        else:
            self.alphabet = pickle.load(open(path, 'r'))
            self.reverse_alphabet = None
            self.num_chars = len(self.alphabet)


    def process_embeddings(self, embeddings_path):
        """
        :param vocab_list: [vocab]
        :return:
        """
        if not gfile.Exists(embeddings_path):
            embeddings = np.diag(np.ones(len(self.alphabet)))
            np.savez_compressed(embeddings_path, embedding=embeddings)
            print("saved one-hot char embeddings matrix at: {}".format(embeddings_path))
            self.embeddings = embeddings

    def get_value(self, char):
        if char in self.alphabet:
            return self.alphabet[char]
        else:
            return self.alphabet[self.unk]

    def get_indices(self, text):
        return [self.get_value(ch) for ch in text]

