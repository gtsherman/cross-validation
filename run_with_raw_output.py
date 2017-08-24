#!/usr/bin/python3

import os

from cross_validation import KFoldValidator
from run import get_args, load_data


class RawResultKFoldValidator(KFoldValidator):
    def __init__(self, raw_dir, num_folds=10, verbose=False):
        self.raw_dir = raw_dir
        super().__init__(num_folds=num_folds, verbose=verbose)

    def test(self, testing, parameters):
        return {item.name: self.read_trec_output(item.name, parameters) for item in testing}

    def read_trec_output(self, query, parameters):
        infile = os.path.join(self.raw_dir, parameters)
        with open(infile) as f:
            raw_results = [line for line in f if query == line.split()[0]]
        return ''.join(raw_results).strip()


def main(args):
    reader = load_data(args)

    # Run cross-validation
    validator = RawResultKFoldValidator(args.raw_dir, num_folds=args.folds, verbose=args.verbose)
    scored_test_items = validator.cross_validate(*reader.scored_items(), seed=args.seed)

    for item in scored_test_items:
        print(scored_test_items[item])


if __name__ == '__main__':
    args = get_args(raw_dir={
        'help': 'the directory containing the raw results (as opposed to the scored results)',
        'required': True
    })

    main(args)
