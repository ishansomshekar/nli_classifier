from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pickle

from tensorflow.python.platform import gfile
import numpy as np
from os.path import join as pjoin

# module_home = os.environ['NLI_PATH']
module_home = '/Users/Chip/dev/cs224s/nli_classifier'
sys.path.insert(0, module_home)

from utils.annotator import Annotator
from utils.data_utils import return_files


class PosEmbeddingWrapper(object):
    def __init__(self):
        self.name = 'pos'
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
            wordcounter = 0
            file_count = 0
            vocab[self.pad] = len(vocab)
            reverse_vocab += [self.pad]
            for file in return_files(self.data_path):
                with open(file, 'r') as f:
                    for i, line in enumerate(f):
                        pos_tags = self.annotator.annotate_pos(line)
                        if len(pos_tags) != len(line.split()):
                            print(pos_tags)
                            print(line.split())
                            assert False

                        for pos in pos_tags:
                            wordcounter += 1
                            if not pos in vocab:
                                vocab[pos] = len(vocab)
                                reverse_vocab += [pos]
                    file_count+=1
                    if file_count % 10 == 0:
                        print ("finished reading %d transcripts" % file_count)

            vocab[self.unk] = len(vocab) 
            reverse_vocab += [self.unk]
            wordcounter += 2

            self.vocab = vocab
            self.reverse_vocab = reverse_vocab
            self.num_tokens = wordcounter
            print( "finished building POS vocabulary of %d words %d tokens for all files" % (len(vocab), wordcounter))
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
        pos_vals = [self.get_value(pos) for pos in pos_tags]
        if (len(pos_tags) != len(pos_vals)):
            print("not same")
            print(pos_tags)
        return pos_vals

    def get_value(self, word):
        if word in self.vocab:
            return self.vocab[word]
        else:
            return self.vocab[self.unk]
