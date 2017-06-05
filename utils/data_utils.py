from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import errno
import sys
import csv
import shutil
from random import shuffle

import numpy as np
from six.moves.urllib.request import urlretrieve
from six.moves import xrange as range
import tensorflow as tf

"""
Data Utils: holds model independent common utility functions for the task
"""

lang_dict = {
    'HIN' : 0,
    'ARA' : 1,
    'JPN' : 2,
    'SPA' : 3,
    'TUR' : 4,
    'GER' : 5,
    'TEL' : 6,
    'KOR' : 7,
    'ITA' : 8,
    'CHI' : 9,
    'FRE' : 10
}

def pad_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    '''Pads each sequence to the same length: the length of the longest
    sequence.
        If maxlen is provided, any sequence longer than maxlen is truncated to
        maxlen. Truncation happens off either the beginning or the end
        (default) of the sequence. Supports post-padding (default) and
        pre-padding.

        Args:
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger
            than maxlen either in the beginning or in the end of the sequence
            value: float, value to pad the sequences to the desired value.
        Returns
            x: numpy array with dimensions (number_of_sequences, maxlen)
            lengths: numpy array with the original sequence lengths
    '''
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths, maxlen


def make_batches(batch_size, data):
    batches = []
    new_order = np.random.permutation(data['labels'].shape[0])
    all_inputs = data['inputs'][new_order]
    all_lens = data['seq_lens'][new_order]
    all_labels = data['labels'][new_order]


    for i in range(0, all_labels.shape[0], batch_size):
        inputs = all_inputs[i:i + batch_size]
        seq_lens = all_seq_lens[i:i + batch_size]
        labels = all_labels[i:i + batch_size]
        batches.append((inputs, seq_lens, labels))

    return batches

def return_files(path):
    return [path+f for f in sorted(os.listdir(path)) if (not f.startswith('missing_files') and not f.startswith('.'))]

def build_data_partition(paths, embedding_wrappers):
    if len(embedding_wrappers) > 1:
        _build_multi_data_partition(paths, embedding_wrappers)
    else:
        _build_single_data_partition(paths, embedding_wrappers[0])


def _build_multi_data_partition(paths, embedding_wrappers):
    inputs_in = paths['inputs_in']
    labels_in = paths['labels_in']
    multi_inputs_out = paths['inputs_out']
    seq_lens_out = paths['seq_lens_out']
    labels_out = paths['labels_out']
    max_len_out = paths['max_len_out']

    assert (len(multi_inputs_out) == len(embedding_wrappers))

    seq_lens = 0
    max_len = 0

    for out_path, ew in zip(multi_inputs_out, embedding_wrappers):
        dataset = []
        for file in return_files(inputs_in):
            with open(file, 'r') as f:
                text = f.read()
                idxs = ew.get_indices(text)
                if len(idxs) > 0 :
                    assert max(idxs) < len(ew.vocab)
            dataset.append(idxs)
        print("path: %s" % out_path)
        res, seq_lens, maxlen = pad_sequences(dataset)
        print("max res")
        print(int(np.array(res).max()))
        with open(out_path, 'wb') as v:
            pickle.dump(res, v)
            print("Saved embeddings at %s" % out_path)

    arr = []
    with open(labels_in, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            arr.append([1 if lang_dict[row[3]] == i else 0 for i in range(len(lang_dict))])

    arr = np.asarray(arr)

    assert(len(seq_lens) == len(labels))

    with open(seq_lens_out, 'w') as v:
        pickle.dump(seq_lens, v)

    with open(labels_out, 'wb') as v:
        pickle.dump(arr, v)

    with open(max_len_out, 'wb') as v:
        pickle.dump(maxlen, v)

def _build_single_data_partition(paths, embedding_wrapper):
    inputs_in = paths['inputs_in']
    labels_in = paths['labels_in']
    inputs_out = paths['inputs_out']
    seq_lens_out = paths['seq_lens_out']
    labels_out = paths['labels_out']
    max_len_out = paths['max_len_out']
    dataset = []
    for file in return_files(inputs_in):
        idxs = []
        with open(file, 'r') as f:
            text = f.read()
            idxs = embedding_wrapper.get_indices(text)
        dataset.append(idxs)
    arr = []

    with open(labels_in, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            arr.append([1 if lang_dict[row[3]] == i else 0 for i in range(len(lang_dict))])

    arr = np.asarray(arr)
    res, seq_lens, maxlen = pad_sequences(dataset)

    with open(inputs_out, 'wb') as v:
        pickle.dump(res, v)

    with open(seq_lens_out, 'w') as v:
        pickle.dump(seq_lens, v)

    with open(labels_out, 'wb') as v:
        pickle.dump(arr, v)

    with open(max_len_out, 'wb') as v:
        pickle.dump(maxlen, v)


def load_embedding_data(path):
    data = np.load(path)
    return data["embedding"]

def ensure_dir(path):
    try:
        os.makedirs(path)
    except OSError as ex:
        if ex.errno != errno.EEXIST:
            raise

def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

def load_data(paths, multi_input=False):
    data = {}
    if multi_input:
        all_inputs = []
        for p in paths['inputs_out']:
           inp = pickle.load(open(p, 'rb'))
           all_inputs.append(inp)
        final_inputs = [[all_inputs[p][i] for p in range(len(paths['inputs_out']))] for i in range(len(all_inputs[0]))]
        final_inputs = np.array(final_inputs)
        data['inputs'] = final_inputs
    else:
        data['inputs'] = pickle.load(open(paths['inputs_out'], 'rb'))
    data['seq_lens'] = pickle.load(open(paths['seq_lens_out'], 'rb'))
    data['labels'] = pickle.load(open(paths['labels_out'], 'rb'))
    data['max_len'] = pickle.load(open(paths['max_len_out'], 'rb'))
    return data

def clear_data(path):
    try:
        shutil.rmtree(path)
    except OSError as ex:
        if ex.errno != errno.ENOENT:
            raise
