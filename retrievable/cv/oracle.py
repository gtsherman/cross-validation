#!/usr/bin/python3

import statistics

from retrievable.cv.cross_validation import KFoldValidator
from run import get_args, load_data


def main(args):
    reader = load_data(args)

    # Run oracle
    scores = []
    validator = KFoldValidator(num_folds=args.folds, verbose=args.verbose)
    for item in reader.scored_items():
        best_params, best_params_score = validator.train([item])
        scores.append(best_params_score)
        if args.verbose:
            print(item.name, best_params, best_params_score, sep='\t')

    if args.summarize:
        print('all', '---', statistics.mean(scores), sep='\t')


if __name__ == '__main__':
    args = get_args()
    main(args)
