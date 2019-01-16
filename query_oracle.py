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
    options.add_argument('--minimize', action='store_true', help='Selects runs having the lowest metric value.')
    args = options.parse_args()

    runs = [read_eval(run, args.qrels, args.metric) for run in args.run]
    optimal = {}
    for query in runs[0]:
        best_score = sorted(runs, key=lambda run: run[query], reverse=not args.minimize)[0][query]
        optimal[query] = best_score

    label_width = max(len(run_name) for run_name in args.run)
    run_names = args.run + ['Optimal']
    for i, run in enumerate(runs + [optimal]):
        print('{run:{lw}} {avg:{lw}}'.format(run=run_names[i], avg=str(statistics.mean(run.values())),
                                             lw=label_width))


def read_eval(file_name, qrels, metric='map'):
    evaluation = {}
    f = subprocess.check_output(['trec_eval', '-q', '-m', metric.lower(), qrels, file_name]).decode('utf-8').strip()
    for line in f.split('\n'):
        _, query, value = line.strip().split()
        evaluation[query] = float(value)
    return evaluation


if __name__ == '__main__':
    main()
