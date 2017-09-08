#!/usr/bin/python3

from cross_validation import RawResultKFoldValidator
from run import get_args, load_data


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
