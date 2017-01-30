import collections
import scored


class ScoreReader:
    def __init__(self):
        self._query_parameter_scores = collections.defaultdict(dict)

    def read(self, file, metric):
        """
        Read a simple key/value file and store data. Assumes file is a tab-separated list of query/value pairs,
        one per line.
        :param file: The file to read
        :param metric: Ignored in this implementation, but relevant to subclasses
        """
        with open(file) as f:
            for line in f:
                query, value = line.strip().split('\t')
                self._query_parameter_scores[query] = value

    def scored_items(self):
        """
        Get the stored query scores as a tuple of Scored instances.
        :return: A list of Scored instances
        """
        return [scored.Scored(query, **self._query_parameter_scores[query]) for query in self._query_parameter_scores]
