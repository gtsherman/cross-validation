# Cross-validation

Generic cross-validation framework intended for TREC-style information retrieval research.

## Usage

The framework is accessed through the `run.py` file. At least one file containing scored items per-parameter setting is required and may be supplied via the `-f` and `-d` options. Typically, the metric to be optimized is also required via the `-m` option; choice of metric depends on your data. See the help for all options:

`./run.py -h`

## Explanation

### Data

#### Data Representation

Data must be labeled for each scored item for each available parameter setting. This means that data should be uniquely identified by its item ID/parameter setting tuple; each such pairing will be associated with some type of score. How this data is ingested can be defined by a custom `score_readers.ScoreReader`, but you can think of the final result as a dictionary of the following form:

```
{
  item_id: {
    parameter_setting: score
  }
}
```

This form is actually encapsulated in the `crossvalidation.Scored` class.

#### Metrics

The framework is agnostic about what each score represents. If your dataset contains more than one type of score for each item/parameter setting tuple, you likely need to specify to your `ScoreReader` which metric's scores should be read. This metric is supplied by the `-m` option to `run.py`.

### Cross-validation

Various forms of cross-validation can generally employ the `cross_validation.KFoldValidator` class. For example, 10-fold cross-validation is the default; leave-one-out cross-validation is possible by setting the number of folds equal to the number of items.

#### Summary Statistics

To identify the optimal parameter setting, the train split must be summarized into a single statistic. In general, the arithmetic mean does a good job summarizing the items' individual scores. However, it might sometimes make sense to optimize a different metric. 

For example, in a [minimax](https://en.wikipedia.org/wiki/Minimax) scenario, it may be preferable to use the _minimum_ score in the training split as the summary statistic. To do so, we simply override the `KFoldValidator` class like so:

```python
class MinimaxKFoldValidator(KFoldValidator):
    def summarize(self, items):
        return min(items.values())
```

Note that by default the `KFoldValidator` seeks to _maximize_ the summary statistic. If you want to _minimize_ the summary statistic, you should either override the `train` method or convert your summary statistic to a form that should be maximized (e.g. by multiplying by -1).
