import collections
import cross_validation


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
        return [cross_validation.Scored(query, **self._item_parameter_scores[query]) for query in
                self._item_parameter_scores]


class TrecScoreReader(ScoreReader):
    def read(self, file, metric):
        """
        Read a trec_eval file and store query->parameters->value.
        :param file: The file to read
        :param metric: The metric to find in the file
        """
        parameter_id = file.split('/')[-1]
        with open(file) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3 and parts[0].lower() == metric:
                    query, value = parts[1], parts[2].strip()
                    if query == 'all':
                        continue
                    self._item_parameter_scores[query][parameter_id] = float(value)