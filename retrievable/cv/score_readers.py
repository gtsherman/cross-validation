import collections
from retrievable.cv.cross_validation import Scored


class ScoreReader:
    def __init__(self):
        self._item_parameter_scores = collections.defaultdict(dict)
        self._parameter_id = 0

    def read(self, file, metric, parameter_id=None):
        """
        Read a simple key/value file and store data. Assumes file is a whitespace-separated list of item/value pairs,
        one per line.
        :param parameter_id: Optional parameter_id attribute to identify this file. If absent, will increment
        :param file: The file to read
        :param metric: Ignored in this implementation, but relevant to subclasses
        """
        if parameter_id is None:
            parameter_id = self._parameter_id
            self._parameter_id += 1
        self._parse_data(file, metric, parameter_id)

    def _parse_data(self, file, metric, parameter_id):
        with open(file) as f:
            for line in f:
                item, value = line.strip().split()
                self._item_parameter_scores[item][parameter_id] = float(value)

    def scored_items(self):
        """
        Get the stored query scores as a tuple of Scored instances.
        :return: A list of Scored instances
        """
        return [Scored(query, **self._item_parameter_scores[query]) for query in
                self._item_parameter_scores]


class TrecScoreReader(ScoreReader):
    def _parse_data(self, file, metric, parameter_id):
        """
        Read a trec_eval file and store query->parameters->value.
        :param parameter_id: Optional parameter_id attribute to identify this file. If absent, will increment
        :param file: The file to read
        :param metric: The metric to find in the file
        """
        for line in file:
            try:
                line = line.decode('utf-8')
            except AttributeError:
                pass
            parts = line.split()
            if len(parts) >= 3 and parts[0].lower() == metric:
                query, value = parts[1], parts[2].strip()
                if query == 'all':
                    continue
                self._item_parameter_scores[query][parameter_id] = float(value)
