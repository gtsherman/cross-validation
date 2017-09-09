import scipy.stats

from cross_validation import KFoldValidator
from run import get_args, load_data


def main(args):
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

if __name__ == '__main__':
    args = get_args(other_directory={
        'help': 'the second directory to compare results against',
        'action': 'append',
        'required': True
    })

    main(args)
