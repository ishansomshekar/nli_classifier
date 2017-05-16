#!/usr/bin/env python3
"""
This script requires Python 3 and the scikit-learn package. See the README file for more details.
Example invocations:
    Generate features from the tokenized essays and tokenized speech transcriptions and build model on them:

        $ python fusion_baseline.py

    Use pre-existing features saved in svmlight format:

        $ python fusion_baseline.py --essay_training_features ../data/features/essays/train/train-2017-03-27-12.58.11.features \
                                    --essay_test_features ../data/features/essays/dev/dev-2017-03-27-12.58.20.features \
                                    --transcription_training_features ../data/features/speech_transcriptions/train/train-2017-04-18-14.36.27.features \
                                    --transcription_test_features ../data/features/speech_transcriptions/dev/dev-2017-04-18-14.36.31.features
"""
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC
from baseline_util import (get_features_and_labels,
                           display_classification_results,
                           write_predictions_file, write_feature_files,
                           combine_feature_matrices)

BASELINE = 'fusion'


def main():
    p = argparse.ArgumentParser()

    p.add_argument('--train',
                   help='Name of training partition. "train" by default. This '
                        'should be the name of a directory in "../data/essays/"'
                        ' as well as "../data/features/"',
                   default='train')

    p.add_argument('--test',
                   help='Name of the testing partition. "dev" by default. This'
                        ' should be the name of a directory in '
                        '"../data/essays/" as well as "../data/features/"',
                   default='dev')

    p.add_argument('--preprocessor',
                   help='Name of directory with processed essay files. '
                        '"tokenized" by default.',
                   default='tokenized')

    p.add_argument('--essay_training_features',
                   help='Path to file containing precomputed training features.'
                        ' None by default. Should be located in '
                        '../data/features/essays/<train_partition_name>/')

    p.add_argument('--essay_test_features',
                   help='Path to file containing precomputed test features. '
                        'None by default. Should be located in '
                        '../data/features/essays/<test_partition_name>/')

    p.add_argument('--transcription_training_features',
                   help='Path to file containing precomputed training features.'
                        ' None by default. Should be located in '
                        '../data/features/speech_transcriptions'
                        '/<train_partition_name>/')

    p.add_argument('--transcription_test_features',
                   help='Path to file containing precomputed test features. '
                        'None by default. Should be located in '
                        '../data/features/speech_transcriptions'
                        '/<test_partition_name>/')

    p.add_argument('--combined_training_features',
                   help='Path to file containing precomputed combined training'
                        ' features. Should be located in ../data/features'
                        '/fusion/<train_partition_name>')

    p.add_argument('--combined_test_features',
                   help='Path to file containing precomputed combined test '
                        'features. Should be located in '
                        '../data/features/fusion/<test_partition_name>')

    p.add_argument('--feature_outfile_name',
                   help='Custom name, if desired, for output feature files to '
                        'be written to '
                        '../data/features/essays/<train_partition_name>/ and '
                        '../data.features/essays/<test_partition_name>. '
                        'If none provided, feature files will be named using '
                        'the date and time. If precomputed feature files are '
                        'provided, this argument will be ignored.')

    p.add_argument('--predictions_outfile_name',
                   help='Custom name, if desired, for predictions file to be '
                        'written to ../predictions/essays/. If none provided, '
                        'predictions file will be names using the date and '
                        'time.')

    args = p.parse_args()
    training_partition_name = args.train
    test_partition_name = args.test
    preprocessor = args.preprocessor
    essay_train_feature_file = args.essay_training_features
    essay_test_feature_file = args.essay_test_features
    transcription_train_feature_file = args.transcription_training_features
    transcription_test_feature_file = args.transcription_test_features
    combined_train_feature_file = args.combined_training_features
    combined_test_feature_file = args.combined_test_features
    feature_outfile_name = args.feature_outfile_name
    predictions_outfile_name = args.predictions_outfile_name

    #
    # Define Vectorizers and Transformers for both essay data and
    # speech_transcriptions. These will be ignored if you provide paths to
    # pre-computed feature files.
    #
    essay_vectorizer, essay_transformer = CountVectorizer(input="filename"), \
                                          Normalizer()
    speech_vectorizer, speech_transformer = CountVectorizer(input="filename"), \
                                            Normalizer()

    if not (combined_train_feature_file and combined_test_feature_file):
        #
        # Get essay features.
        #
        essay_train_and_test_data = get_features_and_labels(training_partition_name,
                                                            test_partition_name,
                                                            essay_train_feature_file,
                                                            essay_test_feature_file,
                                                            baseline='essays',
                                                            preprocessor=preprocessor,
                                                            vectorizer=essay_vectorizer,
                                                            transformer=essay_transformer)

        essay_train_matrix, \
        essay_encoded_train_labels, \
        essay_original_train_labels = essay_train_and_test_data[0]

        essay_test_matrix, \
        essay_encoded_test_labels, \
        essay_original_test_labels = essay_train_and_test_data[1]

        print("Retrieved essay features.")

        #
        # Get speech features.
        #
        speech_train_and_test_data = get_features_and_labels(
            training_partition_name,
            test_partition_name,
            transcription_train_feature_file,
            transcription_test_feature_file,
            baseline='speech_transcriptions',
            preprocessor=preprocessor,
            vectorizer=speech_vectorizer,
            transformer=speech_transformer)

        speech_train_matrix, \
        speech_encoded_train_labels, \
        speech_original_train_labels = speech_train_and_test_data[0]

        speech_test_matrix, \
        speech_encoded_test_labels, \
        speech_original_test_labels = speech_train_and_test_data[1]

        print("Retrieved speech transcription features.")

        assert (speech_original_train_labels == essay_original_train_labels)
        assert (speech_original_test_labels == speech_original_test_labels)

        #
        # Concatenate (horizontally stack) essay and speech feature matrices.
        #
        combined_train_matrix = combine_feature_matrices(essay_train_matrix,
                                                         speech_train_matrix,
                                                         sparse=True)
        print("Finished combining essay and speech transcription "
              "train matrices.")
        combined_test_matrix = combine_feature_matrices(essay_test_matrix,
                                                        speech_test_matrix,
                                                        sparse=True)
        print("Finished combining essay and speech transcription "
              "test matrices.")
        assert (combined_train_matrix.shape[1] == combined_test_matrix.shape[1])

        combined_encoded_train_labels = essay_encoded_train_labels
        combined_encoded_test_labels = essay_encoded_test_labels

        #
        # Write feature files if not provided.
        #
        if not (essay_train_feature_file and essay_test_feature_file):
            write_feature_files(training_partition_name,
                                feature_outfile_name,
                                'essays',
                                essay_train_matrix,
                                essay_encoded_train_labels)

            write_feature_files(test_partition_name,
                                feature_outfile_name,
                                'essays',
                                essay_test_matrix,
                                essay_encoded_test_labels)

        if not (transcription_train_feature_file and
                transcription_test_feature_file):
            write_feature_files(training_partition_name,
                                feature_outfile_name,
                                'speech_transcriptions',
                                speech_train_matrix,
                                speech_encoded_train_labels)

            write_feature_files(test_partition_name,
                                feature_outfile_name,
                                'speech_transcriptions',
                                speech_test_matrix,
                                speech_encoded_test_labels)

        write_feature_files(training_partition_name,
                            feature_outfile_name,
                            BASELINE,
                            combined_train_matrix,
                            speech_encoded_train_labels)

        write_feature_files(test_partition_name,
                            feature_outfile_name,
                            BASELINE,
                            combined_test_matrix,
                            speech_encoded_test_labels)

    else:
        combined_train_and_test_data = get_features_and_labels(
            training_partition_name,
            test_partition_name,
            combined_train_feature_file,
            combined_test_feature_file)

        combined_train_matrix, \
        combined_encoded_train_labels, \
        combined_original_train_labels = combined_train_and_test_data[0]

        combined_test_matrix, \
        combined_encoded_test_labels, \
        combined_original_test_labels = combined_train_and_test_data[1]

    #
    # Train classifier, make predictions, and display results.
    #
    clf = LinearSVC()
    clf.fit(combined_train_matrix, combined_encoded_train_labels)
    predictions = clf.predict(combined_test_matrix)

    write_predictions_file(predictions, test_partition_name,
                           predictions_outfile_name, BASELINE)
    display_classification_results(combined_encoded_test_labels, predictions)


if __name__ == '__main__':
    main()
