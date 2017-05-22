from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pickle

from tensorflow.python.platform import gfile
import numpy as np
from os.path import join as pjoin

module_home = os.environ['NLI_PATH']
sys.path.insert(0, module_home)

from utils.annotator import Annotator


def return_files(path):
    return [path+f for f in os.listdir(path) if (not f.startswith('missing_files') and not f.startswith('.'))]

def return_dir(path):
    return [path+f for f in os.listdir(path) if (not f.startswith('.'))]

class PosEmbeddingWrapper(object):
    def __init__(self):
        self.vocab = None
        self.reverse_vocab = None
        self.embeddings = None
        self.embedding_dim = 50
        self.num_tokens = 0
        self.unk = "UNK"
        self.pad = "<PAD>"
        self.data_path = pjoin(module_home, 'data/speech_transcriptions/train/tokenized/')
        self.annotator = Annotator()


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
                        pos_tags = self.annotator.annotate_pos(line)
			"""
                        if len(pos_tags) != len(line.split()):
                            print(pos_tags)
                            print(line.split())
                            assert False
			"""

                        for pos in pos_tags:
                            wordcounter += 1
                            if not pos in vocab:
                                vocab[pos] = idx
                                reverse_vocab += [pos]
                                idx += 1
                    file_count+=1
                    if file_count % 10 == 0:
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
            print( "finished building POS vocabulary of size %d for all files" %wordcounter)
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
            print("build POS embeddings")
            embeddings = np.diag(np.ones(len(self.vocab)))
            np.savez_compressed(embeddings_path, embedding=embeddings)
            print("saved POS embeddings matrix at: {}".format(embeddings_path))
            self.embeddings = embeddings


    def get_indices(self, text):
        pos_tags = self.annotator.annotate_pos(text)
        return [self.get_value(pos) for pos in pos_tags]
