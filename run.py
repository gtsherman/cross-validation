#!/usr/bin/python

import argparse
import os
import random

from cross_validation import KFoldValidator
from score_readers import ScoreReader, TrecScoreReader

formats = {
    'tsv': ScoreReader,
    'trec': TrecScoreReader
}

def main(args):
    if args.input_format == 'trec' and not args.metric:
        print('Must specify metric for this input format.')
        exit()

    # Load a list of all the supplied files
    files = []
    if args.file is not None:
        files += args.file
    if args.directory is not None:
        files += [os.path.join(directory, file) for directory in args.directory for file in
                  os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
    if len(files) == 0:
        print('Requires at least one file to run.')
        exit()

    # Read data
    reader = formats[args.input_format]()
    for file in files:
        reader.read(file, args.metric)

    # Run cross-validation
    validator = KFoldValidator(args.folds)
    scored_test_items = validator.cross_validate(*reader.scored_items(), seed=args.seed, verbose=args.verbose)

    # Output results
    for item in scored_test_items:
        print('{}\t{}'.format(item, str(scored_test_items[item])))
    if args.summarize:
        print('all\t{}'.format(str(validator.summarize(scored_test_items))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run cross-validation.')
    parser.add_argument('-f', '--file', action='append', help='a file representing an item to be included in '
                                                              'cross-validation; either this or -d must be specified')
    parser.add_argument('-d', '--directory', action='append', help='a directory of files, each of which is an item to '
                                                                   'be included in cross-validation; either this or '
                                                                   '-f must be specified')
    parser.add_argument('-k', '--folds', help='number of folds to use in cross-validation', default=10)
    parser.add_argument('-r', '--seed', help='set the random seed for item shuffling', default=random.random())
    parser.add_argument('-s', '--summarize', action='store_true', help='include summary statistic')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument('-m', '--metric', help='the metric to optimize for cross-validation; required for some input '
                                               'formats')
    parser.add_argument('-i', '--input-format', choices=formats.keys(), help='input file format')
    args = parser.parse_args()

    main(args)
