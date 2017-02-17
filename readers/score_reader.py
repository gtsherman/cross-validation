import collections

from validation import scored


class ScoreReader:
    def __init__(self):
        self._item_parameter_scores = collections.defaultdict(dict)

    def read(self, file, metric):
        """
        Read a simple key/value file and store data. Assumes file is a whitespace-separated list of item/value pairs,
        one per line.
        :param file: The file to read
        :param metric: Ignored in this implementation, but relevant to subclasses
        """
        parameter_id = file.split('/')[-1]
        with open(file) as f:
            for line in f:
                item, value = line.strip().split()
                self._item_parameter_scores[item][parameter_id] = float(value)

    def scored_items(self):
        """
        Get the stored query scores as a tuple of Scored instances.
        :return: A list of Scored instances
        """
        return [scored.Scored(query, **self._item_parameter_scores[query]) for query in self._item_parameter_scores]
