from score_reader import ScoreReader


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
