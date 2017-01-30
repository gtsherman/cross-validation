class Scored:
    def __init__(self, name, **parameter_scores):
        """
        Store a scored item with scores of each parameter setting.
        :param name: Some name uniquely representing this item.
        :param parameter_scores: A kwargs with parameter id as key and item score for that parameter setting as value
        """
        self.name = name
        self.parameter_scores = parameter_scores