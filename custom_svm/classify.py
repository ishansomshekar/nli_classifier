#encoding = utf8
import os
import pickle
import time
import sys
import math
import codecs
import csv
import json

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.preprocessing import Normalizer
from sklearn.metrics import f1_score
import numpy as np
import scipy.sparse

import spacy

module_home = os.environ['NLI_PATH']

def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

def get_files(path, suffix=""):
    return [path+f for f in sorted(os.listdir(path)) if f.endswith(suffix) and not f.startswith('.')]

train_label_file = os.path.join(module_home, 'data/labels/train/labels.train.csv')
train_feat_file = os.path.join(get_script_path(), 'features/train.npz')
train_processed_dir = os.path.join(get_script_path(), 'processed/train/')
train_in_dir = os.path.join(module_home, 'data/speech_transcriptions/train/tokenized/')
train_ivec_file = os.path.join(module_home, 'data/ivectors/train/ivectors.json')


dev_label_file = os.path.join(module_home, 'data/labels/dev/labels.dev.csv')
dev_feat_file = os.path.join(get_script_path(), 'features/dev.npz')
dev_processed_dir = os.path.join(get_script_path(), 'processed/dev/')
dev_in_dir = os.path.join(module_home, 'data/speech_transcriptions/dev/tokenized/')
dev_ivec_file = os.path.join(module_home, 'data/ivectors/dev/ivectors.json')

test_label_file = os.path.join(module_home, 'data/labels/test/labels.test.csv')
test_feat_file = os.path.join(get_script_path(), 'features/test.npz')
test_processed_dir = os.path.join(get_script_path(), 'processed/test/')
test_in_dir = os.path.join(module_home, 'data/speech_transcriptions/test/tokenized/')


class Progbar(object):
    """
    Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, exact=None, alsoprint=None):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """
        values = values or []
        exact = exact or []
        alsoprint = alsoprint or []

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(math.floor(math.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if isinstance(self.sum_values[k], list):
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])
            for k in alsoprint:
                info += ' - %s: %s' % (k[0], k[1])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=None):
        self.update(self.seen_so_far+n, values)

class DataProcessor:
    def __init__(self):
        print("Loading spacy")
        self.spacy = spacy.load('en')
        print("Spacy loaded")

    def process_all_data(self):
        self._process_data(train_in_dir, train_processed_dir)
        self._process_data(dev_in_dir, dev_processed_dir)
        # self._process_data(test_in_dir, test_processed_dir)

    def _get_speaker_id(self, path):
        return path.split('/')[-1].split('.')[0]

    def _get_out_path(self, in_path, out_dir):
        return os.path.join(out_dir, in_path.split('/')[-1].split('.')[0] + '.dat')

    def _file_iterator(self, dir):
        files = get_files(dir, '.txt')
        for file_path in files:
            with codecs.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                yield f.read()


    def _process_data(self, in_dir, out_dir):
        print("Processing data in: %s" % in_dir)
        print("Processing data to: %s" % out_dir)
        files = get_files(in_dir, '.txt')
        filename_it = (fn for fn in files)
        prog = Progbar(target=len(files))
        count = 0
        for sampleDoc in self.spacy.pipe(self._file_iterator(in_dir)):
            file_path = filename_it.next()
            count += 1
            prog.update(count, alsoprint=[("filename", file_path)])
            speaker_id = self._get_speaker_id(file_path)
            words = []
            lemmas = []
            pos_tags = []
            fine_pos_tags = []
            stop_words = []
            oov_count = 0
            for token in sampleDoc:
                words.append(token.text)
                lemmas.append(token.lemma_)
                pos_tags.append(token.pos_)
                fine_pos_tags.append(token.tag_)
                if token.is_stop:
                    stop_words.append(token.lower_)
                if token.is_oov:
                    oov_count += 1
            sample = {'speaker_id': speaker_id, 'words': words, 'lemmas': lemmas, 'pos_tags': pos_tags, 'fine_pos_tags': fine_pos_tags, 'stop_words': stop_words, 'oov_count': oov_count}
            """
            with open(file_path, 'r') as f:
                print "----------------"
                print f.read()
                print sample
                assert False
            """
            with open(self._get_out_path(file_path, out_dir), 'w+') as out_file:
                pickle.dump(sample, out_file)

class FeatureExtractor:
    def __init__(self):
        train_ivecs = json.load(open(train_ivec_file))
        dev_ivecs = json.load(open(dev_ivec_file))
        train_ivecs.update(dev_ivecs)
        self.ivecs = train_ivecs
        self.feat_extractors = [
                lambda data : {"W_" + w + "_UNI": 1 for w in set(data['words'])}, # Unigrams on words
                lambda data : {"L_" + w + "_UNI": 1 for w in set(data['lemmas'])}, # Unigrams on lemmas
                lambda data : {"L_" + w + "_UNI": 1 for w in set(data['stop_words'])}, # Unigrams on stop_words
                lambda data : {"POS_" + w + "_UNI": 1 for w in set(data['pos_tags'])}, # Unigrams on pos_tags
                #lambda data : {"FINE_POS_" + w + "_UNI": 1 for w in set(data['fine_pos_tags'])}, # Unigrams on pos_tags
                #lambda data : {"COUNT_" + w + "_" + str(data['words'].count(w)) : 1 for w in data['words']}, # Word counts
                #lambda data : {"COUNT_" + w: data['words'].count(w) for w in data['words']}, # Word counts
                #lambda data : {"COUNT_" + w: data['lemmas'].count(w) for w in data['lemmas']}, # Lemma counts
                #lambda data : {"COUNT_" + w: data['pos_tags'].count(w) for w in data['pos_tags']}, # POS tag counts
                lambda data : {"W_" + big[0] + "_" + big[1] + "_BIG": 1 for big in set((w1, w2) for w1, w2 in zip(data['words'], data['words'][1:]))}, # Word bigrams
                lambda data : {"L_" + big[0] + "_" + big[1] + "_BIG": 1 for big in set((w1, w2) for w1, w2 in zip(data['lemmas'], data['lemmas'][1:]))}, # Lemma bigrams
                lambda data : {"POS_" + big[0] + "_" + big[1] + "_BIG": 1 for big in set((w1, w2) for w1, w2 in zip(data['pos_tags'], data['pos_tags'][1:]))}, # POS tag bigrams
                #lambda data : {"FINE_POS_" + big[0] + "_" + big[1] + "_BIG": 1 for big in set((w1, w2) for w1, w2 in zip(data['fine_pos_tags'], data['fine_pos_tags'][1:]))}, # POS tag bigrams
                lambda data : {"L_" + big[0] + "_" + big[1] + "_BIG": 1 for big in set((w1, w2) for w1, w2 in zip(data['stop_words'], data['stop_words'][1:]))}, # Stop word bigrams
                #lambda data : {"OOV_COUNT_" + str(data['oov_count']): 1}, # OOV Word count
                #lambda data: {"IVEC_ROW_" + str(i) + ":": self.ivecs[data['speaker_id']][i] for i in xrange(len(self.ivecs[data['speaker_id']]))},
        ]
        self.vectorizer = DictVectorizer(sparse=True)

    def extract_all_features(self):
        self._extract_features(train_processed_dir, train_feat_file, True)
        self._extract_features(dev_processed_dir, dev_feat_file)

    def _extract_features(self, in_dir, out_file, learn=False):
        files = get_files(in_dir, '.dat')
        prog = Progbar(target=len(files))
        count = 0
        feat_dicts = []
        for file_path in files:
            count += 1
            prog.update(count, alsoprint=[("filename", file_path)])
            data_dict = pickle.load(open(file_path, 'r'))
            feats = {}
            for extractor in self.feat_extractors:
                feats.update(extractor(data_dict))
            feat_dicts.append(feats)
        featVec = None
        if learn:
            featVec = self.vectorizer.fit_transform(feat_dicts)
        else:
            featVec = self.vectorizer.transform(feat_dicts)
        scipy.sparse.save_npz(out_file, featVec)

class Classifier:
    def __init__(self):
        self.svc = LinearSVC()
        self.normalizer = Normalizer()
        self.le = LabelEncoder()
        self.train_feats = self._load_all_feats(train_feat_file)
        self.train_labels = self._load_labels(train_label_file)
        self.dev_feats = self._load_all_feats(dev_feat_file)
        self.dev_labels = self._load_labels(dev_label_file)
        self.normalizer.fit_transform(self.train_feats)
        self.normalizer.fit(self.dev_feats)



    def _load_all_feats(self, in_file):
        text_feats = scipy.sparse.load_npz(in_file)



    def _load_labels(self, in_file):
        with open(in_file, 'r') as f:
            reader = csv.reader(f)
            reader.next()
            labels = [row[3] for row in reader]
            return self.le.fit_transform(labels)

    def train(self):
        print("Training SVC")
        self.svc.fit(self.train_feats, self.train_labels)
        print("Training done")

    def eval(self):
        print("Evaluating SVC on dev set")
        preds = self.svc.predict(self.dev_feats)
        f1 = f1_score(self.dev_labels, preds, average=None)
        print("F1: avg: %f" % np.mean(f1))
        print(list(self.le.classes_))
        print(f1)


def main():
    preprocess = False
    feature_process = False
    test = False
    for arg in sys.argv[1:]:
        if arg == "--prep":
            preprocess = True
        if arg == "--feat":
            feature_process = True
        if arg == "--test":
            preprocess = True
    if preprocess:
        processor = DataProcessor()
        processor.process_all_data()
    if feature_process:
        feat_extractor = FeatureExtractor()
        feat_extractor.extract_all_features()
    classifier = Classifier()
    classifier.train()
    classifier.eval()

if __name__ == "__main__":
    main()
