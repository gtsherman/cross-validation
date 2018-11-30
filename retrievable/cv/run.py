#!/usr/bin/python3

import argparse
import os
import random
import tarfile

import scipy.stats

from retrievable.cv.cross_validation import KFoldValidator, RawResultKFoldValidator
from retrievable.cv.score_readers import ScoreReader, TrecScoreReader

formats = {
    'tsv': ScoreReader,
    'trec': TrecScoreReader
}


def main(args=None):
    args = get_args()

    if args.raw_dir:
        main_with_raw_output(args)
    elif args.other_directory:
        main_with_ttest(args)
    else:
        main_default(args)

def main_default(args):
    reader = load_data(args)

    # Run cross-validation
    validator = KFoldValidator(num_folds=args.folds, verbose=args.verbose)
    scored_test_items = validator.cross_validate(*reader.scored_items(), seed=args.seed)

    # Output results
    for item in scored_test_items:
        print('{}\t{}'.format(item, str(scored_test_items[item])))
    if args.summarize:
        print('all\t{}'.format(str(validator.summarize(scored_test_items))))


def main_with_raw_output(args):
    reader = load_data(args)

    # Run cross-validation
    validator = RawResultKFoldValidator(args.raw_dir, num_folds=args.folds, verbose=args.verbose)
    scored_test_items = validator.cross_validate(*reader.scored_items(), seed=args.seed)

    for item in scored_test_items:
        print(scored_test_items[item])


def main_with_ttest(args):
    first_directory = args.directory
    second_directory = args.other_directory

    first_reader = load_data(args)
    args.directory = second_directory
    second_reader = load_data(args)

    # Run cross-validation
    validator = KFoldValidator(num_folds=args.folds, verbose=args.verbose)
    first_scored_test_items = validator.cross_validate(*first_reader.scored_items(), seed=args.seed)
    second_scored_test_items = validator.cross_validate(*second_reader.scored_items(), seed=args.seed)

    # Get the scores in order. Reiterate over the first_scored_test_items even for pulling from
    # second_scored_test_items to ensure the order and items match.
    first_items = [first_scored_test_items[item] for item in first_scored_test_items]
    second_items = [second_scored_test_items[item] for item in first_scored_test_items]

    pval = scipy.stats.ttest_rel(first_items, second_items)[1]

    print('\t'.join([first_directory[0], str(validator.summarize(first_scored_test_items))]))
    print('\t'.join([second_directory[0], str(validator.summarize(second_scored_test_items))]))
    print('two-tailed paired t-test p-value:', pval)


def get_args(**extra_args):
    """
    Get command line options necessary to run most cross-validation.
    :param extra_args: Supply each extra argument with all its argparse options as a dict; the argument name is the flag
    :return: The parsed command line arguments
    """
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
    parser.add_argument('-i', '--input-format', choices=formats.keys(), help='input file format', default='trec')
    parser.add_argument('--raw-dir', help='the directory containing the raw results (as opposed to the scored results)')
    parser.add_argument('--other-directory', help='the second directory to compare results against', action='append')

    if extra_args:
        for extra_arg in extra_args:
            flag = extra_arg.replace('_', '-')
            parser.add_argument('--{}'.format(flag), **extra_args[extra_arg])

    args = parser.parse_args()

    return args


def load_data(args):
    if args.input_format == 'trec' and not args.metric:
        print('Must specify metric for this input format.')
        exit()

    reader = formats[args.input_format]()

    # Load a list of all the supplied files
    if args.file is not None:
        for filepath in args.file:
            if os.path.isfile(filepath):
                with open(filepath) as f:
                    reader.read(f, args.metric, parameter_id=filepath.split('/')[-1])
    if args.directory is not None:
        for directory in args.directory:
            if os.path.isdir(directory):
                for file in os.listdir(directory):
                    filepath = os.path.join(directory, file)
                    if os.path.isfile(filepath):
                        with open(filepath) as f:
                            reader.read(f, args.metric, parameter_id=filepath.split('/')[-1])
            elif tarfile.is_tarfile(directory):
                archive = tarfile.open(directory)
                for member in archive.getmembers():
                    f = archive.extractfile(member)
                    if f is not None:
                        reader.read(f, args.metric, parameter_id=member.name.split('/')[-1])
                archive.close()
    if len(reader.scored_items()) == 0:
        print('Requires at least one file to run.')
        exit()

    return reader


if __name__ == "__main__":
    main()
