#!/usr/bin/env python3
"""
Baseline combining speech transcription features with ivector features.

This script requires Python 3 and the scikit-learn package. See the README file
for more details.

Example invocations:
    Generate the features from the tokenized essays:
        $ python speech_transcriptions+ivectors_baseline.py [--train ] [--test] [--preprocessor]

    Run with precomputed speech transcription features:
        $ python speech_transcriptions+ivectors_baseline.py [--train] [--test dev] [--preprocessor] \
                                                            --transcription_training_features path/to/train/featurefile \
                                                            --transcription_test_features /path/to/test/featurefile
    Run with precomputed combined features:
        $ python speech_transcriptions+ivectors_baseline.py [--train] [--test dev] [--preprocessor] \
                                                            --combined_training_features path/to/train/featurefile \
                                                            --combined_test_features /path/to/test/featurefile
"""

import json
import argparse
from scipy.sparse import csr_matrix
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from baseline_util import (get_features_and_labels,
                           write_predictions_file,
                           display_classification_results,
                           write_feature_files,
                           ivectors_dict_to_features,
                           combine_feature_matrices)

BASELINE = 'speech_transcriptions+ivectors'


def main():
    p = argparse.ArgumentParser()

    p.add_argument('--train',
                   help='Name of training partition. "train" by default. This '
                        'should be the name of a directory in "../data/essays/"'
                        ' as well as "../data/features/"',
                   default='train')

    p.add_argument('--test',
                   help='Name of the testing partition. "dev" by default. This '
                        'should be the name of a directory in "../data/essays/"'
                        ' as well as "../data/features/"',
                   default='dev')

    p.add_argument('--preprocessor',
                   help='Name of directory with processed essay files. '
                        '"tokenized" by default.',
                   default='tokenized')

    p.add_argument('--combined_training_features',
                   help='Path to file containing combined transcription and '
                        'ivector train features')

    p.add_argument('--combined_test_features',
                   help='Path to file containing combined transcription and '
                        'ivector test features')

    p.add_argument('--transcription_training_features',
                   help='Path to file containing precomputed training features.'
                        ' None by default. Should be located in '
                        '../data/features/<train_partition_name>/')

    p.add_argument('--transcription_test_features',
                   help='Path to file containing precomputed test features. '
                        'None by default. Should be located in '
                        '../data/features/<test_partition_name>/')

    p.add_argument('--feature_outfile_name',
                   help='Custom name, if desired, for output feature files '
                        'to be written to '
                        '../data/features/speech_with_ivectors/<train_partition_name>/ and '
                        '../data.features/speech_with_ivectors/<test_partition_name>. '
                        'If none provided, feature files will be named using '
                        'the date and time. If precomputed feature files are '
                        'provided, this argument will be ignored.')

    p.add_argument('--predictions_outfile_name',
                   help='Custom name, if desired, for predictions file to be '
                        'written to ../predictions/essays/. If none provided, '
                        'predictions file will be names using the date and '
                        'time.')

    args = p.parse_args()
    train_partition_name = args.train
    test_partition_name = args.test
    preprocessor = args.preprocessor
    combined_feature_file_train = args.combined_training_features
    combined_feature_file_test = args.combined_test_features
    transcription_feature_file_train = args.transcription_training_features
    transcription_feature_file_test = args.transcription_test_features
    feature_outfile_name = args.feature_outfile_name
    predictions_outfile_name = args.predictions_outfile_name

    #
    # Define Vectorizer and Trasformer
    #
    vectorizer = CountVectorizer(input="filename")
    transformer = Normalizer()  # Normalize frequencies to unit length

    #
    # Load the training and test features and labels
    #
    if not (combined_feature_file_train and combined_feature_file_test):
        transcription_data = get_features_and_labels(train_partition_name,
                                                     test_partition_name,
                                                     transcription_feature_file_train,
                                                     transcription_feature_file_test,
                                                     baseline='speech_transcriptions',
                                                     preprocessor=preprocessor,
                                                     vectorizer=vectorizer,
                                                     transformer=transformer)

        transcription_train_matrix, encoded_train_labels, \
            original_training_labels = transcription_data[0]
        transcription_test_matrix, encoded_test_labels, \
            original_test_labels = transcription_data[1]
        print("Loaded transcription features.")

        ivectors_path = ('../data/features/ivectors/{partition}/'
                         'ivectors.json')
        train_path = ivectors_path.format(partition=train_partition_name)
        train_ivectors_dict = json.load(open(train_path))
        ivectors_train_matrix = ivectors_dict_to_features(train_ivectors_dict,
                                                          train_partition_name,
                                                          mat_format=csr_matrix)

        combined_train_features = combine_feature_matrices(transcription_train_matrix,
                                                           ivectors_train_matrix,
                                                           sparse=True)
        print("Combined transcription features with ivectors for {}"
              .format(train_partition_name))

        test_path = ivectors_path.format(partition=test_partition_name)
        test_ivectors_dict = json.load(open(test_path))

        ivectors_test_matrix = ivectors_dict_to_features(test_ivectors_dict,
                                                         test_partition_name,
                                                         mat_format=csr_matrix)

        combined_test_features = combine_feature_matrices(transcription_test_matrix,
                                                          ivectors_test_matrix,
                                                          sparse=True)

        print("Combined transcription features with ivectors for {}"
              .format(test_partition_name))

        #
        # Write speech transcription features to files if they do not yet exist
        #
        if not (transcription_feature_file_train and
                    transcription_feature_file_test):
            write_feature_files(train_partition_name,
                                feature_outfile_name,
                                'speech_transcriptions',
                                transcription_train_matrix,
                                encoded_train_labels)

            write_feature_files(test_partition_name,
                                feature_outfile_name,
                                'speech_transcriptions',
                                transcription_test_matrix,
                                encoded_test_labels)

        #
        # Write combined transcription + ivector features to file
        #
        write_feature_files(train_partition_name, feature_outfile_name, BASELINE,
                            combined_train_features, encoded_train_labels)
        write_feature_files(test_partition_name, feature_outfile_name, BASELINE,
                            combined_test_features, encoded_test_labels)

    else:
        combined_train_test_data = get_features_and_labels(train_partition_name,
                                                           test_partition_name,
                                                           combined_feature_file_train,
                                                           combined_feature_file_test,
                                                           baseline=BASELINE)
        combined_train_features, encoded_train_labels, \
            original_training_labels = combined_train_test_data[0]
        combined_test_features, encoded_test_labels, \
            original_test_labels = combined_train_test_data[1]
    #
    # Train classifier and predict
    #
    clf = LinearSVC()
    print("Training the classifier...")
    clf.fit(combined_train_features, encoded_train_labels)  # Linear kernel SVM
    predicted = clf.predict(combined_test_features)

    #
    # Write predictions and display report
    #
    write_predictions_file(predicted,
                           test_partition_name,
                           predictions_outfile_name,
                           BASELINE)
    display_classification_results(encoded_test_labels, predicted)


if __name__ == '__main__':
    main()
