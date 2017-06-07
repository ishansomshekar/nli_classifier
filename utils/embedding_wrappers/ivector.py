from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pickle
from os.path import join as pjoin
import json

from tensorflow.python.platform import gfile
import numpy as np

module_home = os.environ['NLI_PATH']
sys.path.insert(0, module_home)

from utils.data_utils import return_files


class IvectorEmbeddingWrapper(object):
    def __init__(self, dim=50):
        self.name = 'ivec'
        self.vocab = None
        self.embeddings = None
        self.num_tokens = 0
        self.unk = "UNK"
        self.pad = "<PAD>"
        self.train_path = pjoin(module_home, 'data/ivectors/train/ivectors.json')
        self.dev_path = pjoin(module_home, 'data/ivectors/dev/ivectors.json')


    def build_vocab(self, path):
        if not gfile.Exists(path):
            print ("building vocabulary for all files")
            dataset_len = 0
            vocab = dict()
            idx = 0
            wordcounter = 0
            file_count = 0
            trainDict = json.load(open(self.train_path))
            devDict = json.load(open(self.dev_path))
            self.fullDict = dict(trainDict, **devDict)
            vocab = {int(speaker_id): idx for idx, speaker_id in enumerate(self.fullDict.keys())}

            idx = len(vocab)
            vocab[self.unk] = idx
            idx += 1
            vocab[self.pad] = idx

            self.vocab = vocab
        else:
            self.vocab = pickle.load(open(path, 'r'))


    def process_embeddings(self, embeddings_path):
        """
        :param vocab_list: [vocab]
        :return:
        """
        if not gfile.Exists(embeddings_path):
            embeddings = np.array(self.fullDict.values())
            np.savez_compressed(embeddings_path, embedding=embeddings)
            self.embeddings = embeddings

    def getIndex(self, speakerId):
        if speakerId in self.vocab:
            return self.vocab[speakerId]
        else:
            return self.vocab[self.unk]
