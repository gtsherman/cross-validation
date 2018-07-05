import collections
import math
import os
import random
import sys
import tarfile


class KFoldValidator:
    def __init__(self, num_folds=10, verbose=False):
        self.num_folds = num_folds
        self.stderr = self.verbose_stderr if verbose else self.silent_stderr

    @staticmethod
    def verbose_stderr(output):
        sys.stderr.write('{}\n'.format(output))

    @staticmethod
    def silent_stderr(output): pass

    def cross_validate(self, *scoreds, **kwargs):
        """
        Run cross-validation.

        Split items into folds, determine optimal parameters based on training items for each fold, and then score
        test items for each fold. Combine each fold of test items to create a complete test run.
        :param scoreds: Scored instances for each item to use for training/testing
        :param kwargs: Optional kwargs specifying:
         1. the shuffle seed; must be a number with keyword "seed"
         2. whether to concatenate the test folds; must be a boolean with keyword "concatenate"; default True
        :return: A dictionary of per-query test scores
        """
        scoreds = self.shuffle(kwargs.get('seed', random.random()), *scoreds)
        folds = self.partition(*scoreds)

        test_scores = self.train_then_test(folds, kwargs.get('concatenate', True))

        return test_scores

    def shuffle(self, seed, *scoreds):
        # Sort the list so that we have a fair comparison given identical seed
        scoreds = sorted(scoreds, key=lambda s: s.name)

        # Shuffle
        random.seed(seed)
        random.shuffle(scoreds)

        return scoreds

    def partition(self, *scoreds):
        """
        Partition the dataset into folds. By default, use equal size folds. Override to partition differently.
        :param scoreds: Scored instances for each item
        :return: A list of lists, each sublist representing a fold
        """
        num_per_chunk = int(max(1, math.ceil(len(scoreds) / float(self.num_folds))))
        folds = [scoreds[i:i + num_per_chunk] for i in range(0, len(scoreds), num_per_chunk)]

        self.stderr('Split into {} folds'.format(str(len(folds))))
        self.stderr('Items per fold: {}'.format(str(num_per_chunk)))

        return folds

    def train_then_test(self, folds, concatenate):
        """
        Run the training and testing procedure. This involves testing on each individual fold after training on
        the remaining folds. Override for different procedure, e.g. train/test split.
        :param folds: The folds of scored items
        :param concatenate: Boolean indicating whether to concatenate the test folds into a unified test output
        :return: A dict of test items and their scores; of the form:
            - name->score, if concatenate is True
            - fold->name->score, if concatenate is False
        """
        test_scores = {}
        for i, fold in enumerate(folds):
            # Get best parameters based on training folds
            training = [scorable for f in folds if f != fold for scorable in f]
            best_parameters, best_score = self.train(training)

            self.stderr('Best params for fold {} (n={}): {} ({})'.format(str(i), str(len(fold)),
                                                                         str(best_parameters), str(best_score)))

            # Store scores for those parameters from test fold
            scores = self.test(fold, best_parameters)
            if concatenate:
                for item_name in scores:
                    test_scores[item_name] = scores[item_name]
            else:
                test_scores[i] = scores

        return test_scores

    def train(self, scoreds):
        """
        Determine the optimal parameter setting for the supplied items.
        :param scoreds: A list or tuple of Scored items
        :return: The parameter key that yields the best overall score
        """

        # Basically inverts the item->param->score structure to a dict of param->name->score
        parameter_items = collections.defaultdict(dict)
        for item in scoreds:
            for parameters in item.parameter_scores:
                parameter_items[parameters][item.name] = item.parameter_scores[parameters]

        # Summarizes each bucket of scores per parameter setting and finds the highest summary score
        summarized_scores = {params: self.summarize(items) for (params, items) in parameter_items.items()}
        best_parameters = max(summarized_scores.keys(), key=lambda params: summarized_scores[params])
        return best_parameters, summarized_scores[best_parameters]

    def test(self, testing, parameters):
        """
        Score the test items using previously determined parameters
        :param testing: A list or tuple of Scored items
        :param parameters: An identifier for the parameters to look up
        :return: An item name->score dict for each testing item
        """
        return {item.name: item.parameter_scores.get(parameters, 0) for item in testing}

    def summarize(self, items):
        """
        By default, calculate the arithmetic mean. Override to use a different summary function.
        :param items: A dict of name->score, each representing a scored item
        :return: A single summary value, in this case the arithmetic mean
        """
        return float(sum(items.values())) / max(len(items), 1)


class RawResultKFoldValidator(KFoldValidator):
    """Gives the optimal raw results for each fold, rather than the per-fold scores"""

    def __init__(self, raw_dir, num_folds=10, verbose=False):
        self._raw_dir = raw_dir
        super().__init__(num_folds=num_folds, verbose=verbose)

    def test(self, testing, parameters):
        raw_dir = self._raw_dir
        if not os.path.isdir(raw_dir) and tarfile.is_tarfile(raw_dir):
            raw_dir = tarfile.open(raw_dir)

        raw_results = {item.name: self._read_output(raw_dir, item.name, parameters) for item in testing}

        try:
            raw_dir.close()
        except AttributeError:
            pass

        return raw_results

    def _read_output(self, raw_dir, item, parameters):
        """
        Reads the raw output files for each parameter ID. This implementation assumes the item name occurs first in
        each line of the raw results, separated with whitespace, as is the case in TREC output format.
        :param item: The identifier, e.g. query number, of the block of raw results needed from the file
        :param parameters: The identifier for the parameters, assuming that each file is named by its parameter setting
        :return: The string containing the raw results for the particular item/parameter tuple
        """
        try:
            infile = raw_dir.extractfile(parameters)
            if infile is not None:
                raw_results = []
                for line in infile:
                    line = line.decode('utf-8')
                    if item == line.split()[0]:
                        raw_results.append(line)
        except AttributeError:  # not a tar file
            infile = os.path.join(raw_dir, parameters)
            with open(infile) as f:
                raw_results = [line for line in f if item == line.split()[0]]
        return ''.join(raw_results).strip()


class Scored:
    def __init__(self, name, **parameter_scores):
        """
        Store a scored item with scores of each parameter setting.
        :param name: Some name uniquely representing this item.
        :param parameter_scores: A kwargs with parameter id as key and item score for that parameter setting as value
        """
        self.name = name
        self.parameter_scores = parameter_scores
