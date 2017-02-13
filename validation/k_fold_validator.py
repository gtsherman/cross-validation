import collections
import random
import sys


class KFoldValidator:

    def __init__(self, num_folds=10):
        self.num_folds = num_folds

    def cross_validate(self, *scoreds, **kwargs):
        """
        Run cross-validation.

        Split items into folds, determine optimal parameters based on training items for each fold, and then score
        test items for each fold. Combine each fold of test items to create a complete test run.
        :param scoreds: Scored instances for each item to use for training/testing
        :param kwargs: Optional kwargs specifying:
         1. the shuffle seed; must be a number with keyword "seed"
         2. whether to include verbose output; must be a boolean with keyword "verbose"
        :return: A dictionary of per-query test scores
        """
        def stderr(output):
            if kwargs['verbose']:
                sys.stderr.write('{}\n'.format(output))
            else:
                return

        scoreds = list(scoreds)
        random.seed(kwargs.get('seed', random.random()))
        random.shuffle(scoreds)

        num_per_chunk = max(1, len(scoreds) / self.num_folds)
        folds = [scoreds[i:i + num_per_chunk] for i in xrange(0, len(scoreds), num_per_chunk)]

        stderr('Split into {} folds'.format(str(len(folds))))
        stderr('Items per fold: {}'.format(str(num_per_chunk)))

        test_scores = {}
        for i,fold in enumerate(folds):
            # Get best parameters based on training folds
            training = [scorable for f in folds if f != fold for scorable in f]
            best_parameters = self.train(training)

            stderr('Best params for fold {}: {}'.format(str(i), str(best_parameters)))

            # Store scores for those parameters from test fold
            scores = self.test(fold, best_parameters)
            for item_name in scores:
                test_scores[item_name] = scores[item_name]

        return test_scores

    def train(self, scoreds):
        """
        Determine the optimal parameter setting for the supplied items.
        :param scoreds: A list or tuple of Scored items
        :return: The parameter key that yields the best overall score
        """

        # Basically just combines the parameter->score dicts of each item into a big parameter->[score,score,...] dict
        parameter_scores = collections.defaultdict(list)
        for item in scoreds:
            for parameters in item.parameter_scores:
                parameter_scores[parameters].append(item.parameter_scores[parameters])

        # Summarizes each bucket of scores per parameter setting and finds the highest summary score
        summarized_scores = {params: self.summarize(scores) for (params, scores) in parameter_scores.iteritems()}
        best_parameters = max(summarized_scores.iterkeys(), key=lambda params: summarized_scores[params])
        return best_parameters

    @staticmethod
    def test(testing, parameters):
        """
        Score the test items using previously determined parameters
        :param testing: A list or tuple of Scored items
        :param parameters: An identifier for the parameters to look up
        :return: An item name->score dict for each testing item
        """
        return {item.name: item.parameter_scores.get(parameters, 0) for item in testing}

    @staticmethod
    def summarize(scores):
        """
        By default, calculate the arithmetic mean. Override to use a different summary function.
        :param scores: A list or tuple of numbers to summarize
        :return: A single summary value, in this case the arithmetic mean
        """
        return float(sum(scores)) / max(len(scores), 1)
