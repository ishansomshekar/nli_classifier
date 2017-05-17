#!/usr/bin/env python3

"""
This script requires Python 3 and the scikit-learn package. See the README file for more details.
Example invocations:
    Generate the features from the tokenized essays:
        $ python speech_transcription_baseline.py [--train ] [--test] [--preprocessor]

    Run with precomputed features:
        $ python speech_transcription_baseline.py [--train] [--test dev] [--preprocessor] \
                                                  --training_features path/to/train/featurefile \
                                                  --test_features /path/to/test/featurefile
"""
import argparse
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from baseline_util import (get_features_and_labels, write_predictions_file,
                           display_classification_results, write_feature_files)

BASELINE = 'speech_transcriptions'


def main():
    p = argparse.ArgumentParser()

    p.add_argument('--train',
                   help='Name of training partition. "train" by default. This should be the name of a directory '
                        'in "../data/essays/" as well as "../data/features/"',
                   default='train')
    p.add_argument('--test',
                   help='Name of the testing partition. "dev" by default. This should be the name of a directory '
                        'in "../data/essays/" as well as "../data/features/"',
                   default='dev')
    p.add_argument('--preprocessor',
                   help='Name of directory with processed essay files. "tokenized" by default.',
                   default='tokenized')
    p.add_argument('--training_features',
                   help='Path to file containing precomputed training features. None by default. '
                        'Should be located in ../data/features/<train_partition_name>/')
    p.add_argument('--test_features',
                   help='Path to file containing precomputed test features. None by default.'
                        'Should be located in ../data/features/<test_partition_name>/')
    p.add_argument('--feature_outfile_name',
                   help='Custom name, if desired, for output feature files to be written to '
                        '../data/features/essays/<train_partition_name>/ and '
                        '../data.features/essays/<test_partition_name>. '
                        'If none provided, feature files will be named using the date and time.'
                        'If precomputed feature files are provided, this argument will be ignored.')
    p.add_argument('--predictions_outfile_name',
                   help='Custom name, if desired, for predictions file to be written to ../predictions/essays/.'
                        'If none provided, predictions file will be names using the date and time.')
    args = p.parse_args()
    train_partition_name = args.train
    test_partition_name = args.test
    preprocessor = args.preprocessor
    feature_file_train = args.training_features
    feature_file_test = args.test_features
    feature_outfile_name = args.feature_outfile_name
    predictions_outfile_name = args.predictions_outfile_name

    #
    # Define Vectorizer and Transformer
    #
    vectorizer = CountVectorizer(input="filename")
    transformer = Normalizer()  # Normalize frequencies to unit length

    #
    # Load the training and test features and labels
    #
    training_and_test_data = get_features_and_labels(train_partition_name,
                                                     test_partition_name,
                                                     feature_file_train,
                                                     feature_file_test,
                                                     baseline=BASELINE,
                                                     preprocessor=preprocessor,
                                                     vectorizer=vectorizer,
                                                     transformer=transformer)

    train_matrix, encoded_train_labels, original_training_labels = training_and_test_data[0]
    test_matrix, encoded_test_labels, original_test_labels = training_and_test_data[1]

    #
    # Write features to feature files if they are new
    #
    if not (feature_file_train and feature_file_test):
        write_feature_files(train_partition_name, feature_outfile_name, BASELINE, train_matrix, encoded_train_labels)
        write_feature_files(test_partition_name, feature_outfile_name, BASELINE, test_matrix, encoded_test_labels)

    #
    # Run the classifier
    #
    clf = LinearSVC()
    print("Training the classifier...")
    clf.fit(train_matrix, encoded_train_labels)  # Linear kernel SVM
    predicted = clf.predict(test_matrix)

    #
    # Write predictions and display report
    #
    write_predictions_file(predicted, test_partition_name, predictions_outfile_name, BASELINE)
    display_classification_results(encoded_test_labels, predicted)


if __name__ == '__main__':
    main()