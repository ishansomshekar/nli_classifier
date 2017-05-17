#!/usr/bin/env python3
"""
This script requires Python 3 and the scikit-learn package. See the README file 
for more details.

Example invocations:
    Generate features from the tokenized essays and tokenized speech 
    transcriptions and build model on them:

        $ python fusion+ivectors_baseline.py [--train ] [--test] [--preprocessor]

    Use precomputed essay/speech transciption features saved in svmlight format:

        $ python fusion+ivectors_baseline.py [--train ] [--test] [--preprocessor] \
                                             --essay_training_features relpath/to/featuresfile \
                                             --essay_test_features relpath/to/featuresfile \
                                             --transcription_training_features relpath/to/featuresfile \
                                             --transcription_test_features  relpath/to/featuresfile
    Use precomputed combined feature files:
        $ python fusion+ivectors_baseline.py --combined_training_features \
                                                    relpath/to/featuresfile \
                                             --combined_test_features \
                                                    relapth/to/featuresfile
"""
import argparse
import json
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC
from baseline_util import (get_features_and_labels,
                           display_classification_results,
                           write_predictions_file,
                           write_feature_files,
                           combine_feature_matrices,
                           ivectors_dict_to_features)

BASELINE='fusion+ivectors'


def main():
    p = argparse.ArgumentParser()

    p.add_argument('--train',
                   help='Name of training partition. "train" by default. '
                        'This should be the name of a directory in '
                        '"../data/essays/" as well as "../data/features/"',
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
                        '../data/features/speech_transcriptions/'
                        '<train_partition_name>/')
    
    p.add_argument('--transcription_test_features',
                   help='Path to file containing precomputed test features. '
                        'None by default. Should be located in '
                        '../data/features/speech_transcriptions/'
                        '<test_partition_name>/')
    
    p.add_argument('--combined_training_features',
                   help='Path to file containing precomputed combined training '
                        'features. Should be located in '
                        '../data/features/fusion/<train_partition_name>')
    
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
    train_partition_name = args.train
    test_partition_name = args.test
    preprocessor = args.preprocessor
    essay_train_feature_file = args.essay_training_features
    essay_test_feature_file = args.essay_test_features
    transcription_train_feature_file = args.transcription_training_features
    transcription_test_feature_file = args.transcription_test_features
    combined_feature_file_train = args.combined_training_features
    combined_feature_file_test = args.combined_test_features
    feature_outfile_name = args.feature_outfile_name
    predictions_outfile_name = args.predictions_outfile_name

    if not (combined_feature_file_train and combined_feature_file_test):
        #
        # Define Vectorizers and Transformers for both essay data and
        # speech_transcriptions. These will be ignored if you provide paths
        # to pre-computed feature files.
        #
        essay_vectorizer, \
            essay_transformer = CountVectorizer(input="filename"), Normalizer()
        speech_vectorizer, \
            speech_transformer = CountVectorizer(input="filename"), Normalizer()

        #
        # Get essay features.
        #
        essay_data = get_features_and_labels(train_partition_name,
                                             test_partition_name,
                                             essay_train_feature_file,
                                             essay_test_feature_file,
                                             baseline='essays',
                                             preprocessor=preprocessor,
                                             vectorizer=essay_vectorizer)

        essay_train_matrix, essay_encoded_train_labels, \
            essay_original_train_labels = essay_data[0]

        essay_test_matrix, essay_encoded_test_labels, \
            essay_original_test_labels = essay_data[1]

        essay_train_matrix = essay_transformer.fit_transform(essay_train_matrix)
        essay_test_matrix = essay_transformer.transform(essay_test_matrix)
        print("Retrieved essay features.")

        #
        # Get speech features.
        #
        speech_data = get_features_and_labels(train_partition_name,
                                              test_partition_name,
                                              transcription_train_feature_file,
                                              transcription_test_feature_file,
                                              baseline='speech_transcriptions',
                                              preprocessor=preprocessor,
                                              vectorizer=speech_vectorizer)

        speech_train_matrix, speech_encoded_train_labels, \
            speech_original_train_labels = speech_data[0]

        speech_test_matrix, speech_encoded_test_labels, \
            speech_original_test_labels = speech_data[1]
        print("Retrieved speech transcription features.")

        assert(speech_original_train_labels == essay_original_train_labels)
        assert(speech_original_test_labels == speech_original_test_labels)

        #
        # Load ivectors
        #
        ivectors_path = ('../data/features/ivectors/'
                         '{partition}/ivectors.json')

        train_path = ivectors_path.format(partition=train_partition_name)
        train_ivectors_dict = json.load(open(train_path))
        ivectors_train_matrix = ivectors_dict_to_features(train_ivectors_dict,
                                                          train_partition_name,
                                                          mat_format=csr_matrix)
        test_path = ivectors_path.format(partition=test_partition_name)
        test_ivectors_dict = json.load(open(test_path))
        ivectors_test_matrix = ivectors_dict_to_features(test_ivectors_dict,
                                                         test_partition_name,
                                                         mat_format=csr_matrix)

        #
        # Combine (horizontally stack) essay, speech_transcription, and ivector
        # feature matrices.
        #
        essay_and_trans_train = combine_feature_matrices(essay_train_matrix,
                                                         speech_train_matrix,
                                                         sparse=True)
        combined_train_matrix = combine_feature_matrices(essay_and_trans_train,
                                                         ivectors_train_matrix,
                                                         sparse=True)

        print("Finished combining essay, speech transcription, and ivector "
              "train matrices.")

        essay_and_trans_test = combine_feature_matrices(essay_test_matrix,
                                                        speech_test_matrix,
                                                        sparse=True)
        combined_test_matrix = combine_feature_matrices(essay_and_trans_test,
                                                        ivectors_test_matrix,
                                                        sparse=True)

        print("Finished combining essay, speech transcription, and ivector "
              "test matrices.")

        assert (combined_train_matrix.shape[1] == combined_test_matrix.shape[1])

        encoded_train_labels = essay_encoded_train_labels
        encoded_test_labels = essay_encoded_test_labels

        #
        # Write feature files if not provided.
        #

        if not (essay_train_feature_file and essay_test_feature_file):
            write_feature_files(train_partition_name,
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

            write_feature_files(train_partition_name,
                                feature_outfile_name,
                                'speech_transcriptions',
                                speech_train_matrix,
                                speech_encoded_train_labels)

            write_feature_files(test_partition_name,
                                feature_outfile_name,
                                'speech_transcriptions',
                                speech_test_matrix,
                                speech_encoded_test_labels)

        write_feature_files(train_partition_name,
                            feature_outfile_name,
                            BASELINE,
                            combined_train_matrix,
                            encoded_train_labels)
        write_feature_files(test_partition_name,
                            feature_outfile_name,
                            BASELINE,
                            combined_test_matrix,
                            encoded_test_labels)
    else:
        #
        # Load precomputed combined features
        #
        combined_data = get_features_and_labels(train_partition_name,
                                                test_partition_name,
                                                combined_feature_file_train,
                                                combined_feature_file_test,
                                                baseline=BASELINE)

        combined_train_matrix, encoded_train_labels, \
            original_training_labels = combined_data[0]

        combined_test_matrix, encoded_test_labels, \
            original_test_labels = combined_data[1]

    #
    # Train classifier and predict
    #
    clf = LinearSVC()
    clf.fit(combined_train_matrix, encoded_train_labels)
    predictions = clf.predict(combined_test_matrix)

    #
    # Write predictions and display report
    #
    write_predictions_file(predictions,
                           test_partition_name,
                           predictions_outfile_name,
                           BASELINE)

    display_classification_results(encoded_test_labels,
                                   predictions)

if __name__ == '__main__':
    main()
