import argparse
import statistics
import subprocess


def main():
    options = argparse.ArgumentParser(description='Given one or more TREC-formatted run files, select the optimal run '
                                                  'for each query.')
    options.add_argument('qrels', help='A standard qrels file.')
    options.add_argument('run', nargs='+', help='One or more TREC-formatted run files.')
    options.add_argument('-m', '--metric', default='map', help='The metric to optimize. This should be equivalent to '
                                                               'the -m argument you would pass to trec_eval (which is '
                                                               'not always equivalent to the name displayed for the '
                                                               'metric, e.g. ndcg@20 is displayed as ndcg_cut_20 but '
                                                               'is passed as -m ndcg_cut.20). Defaults to MAP.')
    options.add_argument('-q', '--per-query', action='store_true', help='Instead of the summary evaluation metric for '
                                                                        'each run, print the optimal run and '
                                                                        'evaluation score for each query.')
    options.add_argument('--minimize', action='store_true', help='Selects runs having the lowest metric value.')
    args = options.parse_args()

    runs = {run: read_eval(run, args.qrels, args.metric) for run in args.run}

    run_label_width = max(len(run_name) for run_name in args.run)
    queries = [query_name for query_name in list(runs.values())[0]]

    optimal_scores = {}
    for query in queries:
        best_run = sorted(runs.keys(), key=lambda run_name: runs[run_name][query], reverse=not args.minimize)[0]
        best_score = runs[best_run][query]
        optimal_scores[query] = best_score
        if args.per_query:
            print('{query},{run},{val}'.format(query=query, run=best_run, val=best_score))

    if not args.per_query:
        runs['Optimal'] = optimal_scores
        for run_name in runs:
            print('{run:{lw}} {avg:{lw}}'.format(run=run_name, avg=str(statistics.mean(runs[run_name].values())),
                                                 lw=run_label_width))


def read_eval(file_name, qrels, metric='map'):
    evaluation = {}
    f = subprocess.check_output(['trec_eval', '-q', '-m', metric.lower(), qrels, file_name]).decode('utf-8').strip()
    for line in f.split('\n'):
        _, query, value = line.strip().split()
        if query != 'all':
            evaluation[query] = float(value)
    return evaluation


if __name__ == '__main__':
    main()
