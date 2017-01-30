#!/usr/bin/python

import argparse
import os
import random

from k_fold_validator import KFoldValidator
from trec_score_reader import TrecScoreReader


def main(args):
    # Load a list of all the supplied files
    files = []
    if args.file is not None:
        files += args.file
    if args.directory is not None:
        files += [os.path.join(directory, file) for directory in args.directory for file in
                  os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
    if len(files) == 0:
        print('Requires at least one file to run')
        exit()

    # Read data
    reader = TrecScoreReader()
    for file in files:
        reader.read(file, args.metric)

    # Run cross-validation
    validator = KFoldValidator(args.folds)
    scored_test_items = validator.cross_validate(*reader.scored_items(), seed=args.seed)

    # Output results
    for item in scored_test_items:
        print('{}\t{}'.format(item, str(scored_test_items[item])))
    if args.summarize:
        print('all\t{}'.format(str(validator.summarize(scored_test_items.values()))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run cross-validation.')
    parser.add_argument('-f', '--file', action='append', help='a file representing an item to be included in '
                                                              'cross-validation')
    parser.add_argument('-d', '--directory', action='append', help='a directory of files, each of which is an item to '
                                                                   'be included in cross-validation')
    parser.add_argument('-k', '--folds', help='number of folds to use in cross-validation', default=10)
    parser.add_argument('-m', '--metric', help='the metric to optimize for cross-validation', required=True)
    parser.add_argument('-r', '--seed', help='set the random seed for item shuffling', default=random.random())
    parser.add_argument('-s', '--summarize', action='store_true', help='include summary statistic')
    args = parser.parse_args()

    main(args)
