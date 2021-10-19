#!python3.7
# pylint: disable=line-too-long
# based on DeepLTL: https://github.com/reactive-systems/deepltl

"""Dataset generator for LTL formulas based on rich specification patterns

Invoke module from command line (python -m ...)
Supports Google cloud storage targets"""

import sys, os
import signal
import datetime
import argparse
import random
import json
from typing import Tuple

from tensorflow.io import gfile # pylint: disable=import-error

from tgan_sr.utils import ltl_parser, utils
from tgan_sr.data_generation.ltl import solve_ltl
from tgan_sr.data_generation.spec_patterns import SpecPatternGen



class DistributionGate():
    """Keep track of formula distribution so far"""
    def __init__(self, key, distribution, interval, total_num, **kwargs):
        """
        key : 'formula_size'
        distribution : 'uniform' or 'arbitrary'
        interval : tuple (min,max) of allowed sizes
        total_num : target number of examples
        kwargs (optional): start_calc_at together with alpha, both only for uniform
        """
        self.dist = {}
        self.targets = {}
        self.fulls = {}
        self.key = key
        self.interval = interval
        self.alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.0
        self.distribution = distribution
        bleft, bright = interval
        if key == 'formula size':
            self.bins = list(range(bleft, bright + 1))
            self.get_val = lambda x: x.size()
        else:
            raise ValueError()
        for b in self.bins:
            self.dist[b] = 0
        if distribution == 'uniform':
            if 'start_calc_from' in kwargs:
                start = kwargs['start_calc_from']
                self.enforced_bins = list(
                    filter(lambda x: x >= start, self.bins))
            else:
                self.enforced_bins = self.bins
            num_actual_bins = len(self.enforced_bins)
            for b in self.bins:
                self.targets[b] = total_num * \
                    (1 - self.alpha) / num_actual_bins
                self.fulls[b] = self.dist[b] >= self.targets[b]
        elif distribution == 'arbitrary':
            pass
        else:
            raise ValueError()

    def gate(self, formula_obj: ltl_parser.LTLFormula) -> bool:
        """check if formula matches distribution"""
        val = self.get_val(formula_obj)
        if val < self.interval[0] or val > self.interval[1]:  # not in range
            return False
        if self.distribution == 'arbitrary':
            return True
        else:
            return not self.fulls[val]

    def update(self, formula_obj: ltl_parser.LTLFormula):
        """add formula to distribution"""
        val = self.get_val(formula_obj)
        if val >= self.interval[0] and val <= self.interval[1]:
            self.dist[val] += 1
            if self.distribution != 'arbitrary' and self.dist[val] >= self.targets[val]:
                self.fulls[val] = True

    def histogram(self, show=True, save_to=None):
        """plot the formula distribution so far as histogram"""
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(1)
        counts = [val for key, val in sorted(self.dist.items())]
        axis.bar(self.bins, counts, width=1,
                 color='#3071ff', edgecolor='white')
        axis.set_ylabel('number of items')
        axis.set_xlabel(self.key)
        axis.title.set_text('alpha = ' + str(self.alpha))
        if save_to is not None:
            figure.savefig(save_to)
        if show:
            plt.show()
        else:
            plt.close(figure)

    def full(self) -> bool:
        if self.distribution == 'arbitrary':
            return False
        else:
            return all([self.fulls[eb] for eb in self.enforced_bins])



def generate_examples(params) -> Tuple[list, list, dict]:
    """main data generation function. return list of examples, optional list of timed-out examples and info dict"""

    interrupted = False
    def signal_handler(signal, frame):
        nonlocal interrupted
        print(f"Received signal {signal:d}, interrupting generation")
        interrupted = True
    signal.signal(signal.SIGINT, signal_handler)

    if params.num_variables > 26:
        raise ValueError("Cannot generate more than 26 APs")
    variables = list(map(chr, range(97, 97 + params.num_variables)))
    if not isinstance(params.tree_size, tuple):
        params.tree_size = (1, params.tree_size)

    class SpecPatternGenWrapper:
        def __next__(self):
            d = SpecPatternGen(variables, params.tree_size)
            return d.run()
    formula_generator = SpecPatternGenWrapper()

    dist_gate = DistributionGate('formula size', params.formula_dist, params.tree_size, params.num_examples, start_calc_from=10, alpha=params.alpha)

    # generate samples
    print('Generating examples...')
    examples = []
    timeout_formulas = []
    sat_examples = 0
    unsat_examples = 0
    total_examples = 0
    dropped_sat_examples = 0
    dropped_unsat_examples = 0
    dropped_dist_examples = 0
    dropped_timeout_examples = 0
    time_started = datetime.datetime.now()
    last_msg_time = time_started
    last_msg_percent = 0
    accu = { k : 0 for k in {'in_length', 'out_length', 'solve_time'}}
    info = {'max_in_length' : 0, 'max_out_length' : 0}
    while True:
        current_percent = total_examples / params.num_examples * 100
        now = datetime.datetime.now()
        if current_percent - last_msg_percent >= params.log_each_x_percent or now - last_msg_time > datetime.timedelta(hours=1):
            last_msg_percent = current_percent
            last_msg_time = now
            print("Generated {:,d} of {:,d} examples ({:4.1f}%); dropped {:,d} sat, {:,d} unsat, {:,d} dist, {:,d} timeout; at {:s} runtime".format(total_examples, 
              params.num_examples, current_percent, dropped_sat_examples, dropped_unsat_examples, dropped_dist_examples, dropped_timeout_examples, utils.strfdelta_hms(now - time_started)))
            sys.stdout.flush()
        if total_examples >= params.num_examples or dist_gate.full() or interrupted:
            break
        if params.max_runtime != 0.0 and (now - time_started).total_seconds() > params.max_runtime:
            print('Exiting due to exceeded runtime')
            break

        formula = next(formula_generator)
        if not isinstance(formula, ltl_parser.LTLFormula):
            if not isinstance(formula, str):
                if formula is None:
                    continue
                formula = formula.to_str()
            formula_obj = ltl_parser.ltl_formula(formula, 'spot')
        else:
            formula_obj = formula

        if not dist_gate.gate(formula_obj):  # formula doesn't fit distribution
            dropped_dist_examples += 1
            continue
        if formula == '1':
            continue

        is_sat, witness, d = solve_ltl(formula_obj, timeout=params.timeout, binary_path=params.binary_dir)

        if is_sat is None:  # due to timeout
            if params.save_timeouts:
                timeout_formulas.append(formula_obj.to_str('spot', spacing='all ops', full_parens=True))
            dropped_timeout_examples += 1
            continue
        elif not is_sat: # unsat
            if (params.frac_unsat is not None) and unsat_examples >= params.frac_unsat * params.num_examples:
                dropped_unsat_examples += 1
                continue
            else:  # more unsat samples needed
                witness_str = '0'
                dist_gate.update(formula_obj)
                unsat_examples += 1
        else:  # is_sat
            if (params.frac_unsat is not None) and sat_examples >= (1 - params.frac_unsat) * params.num_examples:
                dropped_sat_examples += 1
                continue
            elif random.random() < params.drop_sat_prob:
                # don't log
                continue
            else:  # more sat samples needed
                witness_str = '1'
                out_length = len(witness_str)
                info['max_out_length'] = max(info['max_out_length'], out_length)
                accu['out_length'] += out_length
                dist_gate.update(formula_obj)
                sat_examples += 1

        for k, v in d.items(): # update info accumulator with this samples' info
            if k in accu:
                accu[k] += v
        in_length = formula_obj.size()
        info['max_in_length'] = max(info['max_in_length'], in_length)
        accu['in_length'] += in_length
        formula_str = formula_obj.to_str('network-polish')
        examples.append((formula_str, witness_str, d))
        total_examples += 1

    if params.include_solve_time:
        info['avg_solve_time'] = accu['solve_time'] / total_examples
        print('Average solve time: {avg_solve_time:.2f} ms'.format(**info))
    info['avg_in_length'] = accu['in_length'] / total_examples
    info['avg_out_length'] = accu['out_length'] / sat_examples
    info['runtime'] = utils.strfdelta_hms(datetime.datetime.now() - time_started)
    info['_dist_gate'] = dist_gate
    info['examples_generated'] = total_examples
    info['examples_generated_sat'] = sat_examples
    info['examples_generated_unsat'] = unsat_examples
    print('Average formula length {avg_in_length:.1f} and witness length {avg_out_length:.1f}'.format(**info))
    print('Generated {:d} examples ({:d} sat, {:d} unsat). {:d} requested.'.format(total_examples, sat_examples, unsat_examples, params.num_examples))
    return examples, timeout_formulas, info


def split_and_write(examples, timeouts, params, log_dict):
    """writes generated examples to files, acoording to split parameters. Also adds info json and distribution files."""
    random.Random(params.seed).shuffle(examples)
    num_examples = len(examples)
    res = {}
    total_val = sum(params.splits.values())
    current_val = 0.0
    for split, val in params.splits.items():
        res[split] = examples[int(current_val/total_val * num_examples) : int((current_val + val)/total_val * num_examples)]
        current_val += val

    if params.include_solve_time:
        log_dict['file_format'] += '+solve_time'

    print(f'Writing dataset of {num_examples} to {params.output_dir}...')
    gfile.makedirs(params.output_dir)
    for split, data in res.items():
        path = os.path.join(params.output_dir, split + '.txt')
        with gfile.GFile(path, 'w') as f:
            for ex_in, ex_out, d in data:
                f.write(ex_in)
                if params.include_solve_time:
                    f.write(' #solve_time: {solve_time:.2f}'.format(**d))
                f.write('\n' + ex_out + '\n')
    log_dict['_dist_gate'].histogram(show=False, save_to='tmp_dist.png')      # For distribution analysis
    gfile.copy('tmp_dist.png', os.path.join(params.output_dir, 'dist.png'), overwrite=True)
    gfile.remove('tmp_dist.png')
    del log_dict['_dist_gate']
    with gfile.GFile(os.path.join(params.output_dir, 'info.json'), 'w') as f:
        log_dict['timestamp'] = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        f.write(json.dumps(log_dict, indent=4) + '\n')

    if params.save_timeouts:
        with gfile.GFile(os.path.join(params.output_dir, 'timeouts.txt'), 'w') as f:
            f.write('\n\n'.join(timeouts) + '\n\n')


def run():
    # Argument processing
    parser = argparse.ArgumentParser(description='Data generator for LTL specification patterns')
    parser.add_argument('--output-dir', '-od', type=str, required=True, help='dataset output directory. will contain one dataset file per split and various info files.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--splits', default='train:8,val:1,test:1', help='proportions for splitting the generated examples into subsets')
    parser.add_argument('--timeout', type=float, default=10, help='time in seconds to wait for the generation of a single example')
    parser.add_argument('--num-variables', '-nv', type=int, default=5)
    parser.add_argument('--num-examples', '-ne', type=int, default=1000, help='target size of the dataset (before splitting)')
    parser.add_argument('--tree-size', '-ts', type=str, default='15', metavar='MAX_TREE_SIZE', help="Maximum tree size of generated formulas. Range can be specified as 'MIN-MAX'; default minimum is 1")
    parser.add_argument('--formula-dist', type=str, default='arbitrary', help='formula size distribution. arbitrary or uniform. See DistributionGate.')
    parser.add_argument('--alpha', type=float, default=0.0, help='Distribution parameter for uniform.')
    parser.add_argument('--drop-sat-prob', type=float, default=0.0, help='drop satisfiable instances right after generation with this probability. sat/unsat proportions will be filtered regardless, if specified.')
    parser.add_argument('--frac-unsat', type=str, default='0.5', help="target proportions of sat/unsat samples over the whole dataset. can be 'none' to not filter at all.")
    parser.add_argument('--include-solve-time', action='store_true', help='tag examples with the runtime of the used solver')
    parser.add_argument('--binary-dir', default='./bin', help='binary directory for aalta')
    parser.add_argument('--save-timeouts', action='store_true', help='save examples that could not be solved in the specified time to a distinct file')
    parser.add_argument('--max-runtime', type=float, default=0.0, help='total maximum runtime of this script; generation will be stopped if expired')
    parser.add_argument('--log-each-x-percent', type=float, default=1.0, help='percent of target examples intervals for progress updates')
    parser.add_argument('--comment', '-C', type=str, help='will be included in the info json of the dataset')

    args = parser.parse_args()
    original_args = argparse.Namespace(**vars(args))

    if '-' in args.tree_size:
        args.tree_size = tuple(map(int, args.tree_size.split('-')))
    else:
        args.tree_size = int(args.tree_size)
    args.splits = { k : int(v) for k, v in [q.strip().split(':') for q in args.splits.split(',')] }
    if args.frac_unsat.lower() == 'none':
        args.frac_unsat = None
    else:
        args.frac_unsat = float(args.frac_unsat)


    log_dict = vars(original_args)
    log_dict.update({'problem' : 'ltl', 'subproblem' : 'decision', 'operator-notation' : 'polish', 'ltl-solver' : 'aalta', 'file-format' : 'text/2lines',
            'formula-generator' : 'spec_patterns'})
    ld_path = os.environ.get('LD_LIBRARY_PATH', None)
    os.environ['LD_LIBRARY_PATH'] = args.binary_dir + ((':' + ld_path) if ld_path is not None else '') # TODO only works on linux..
    examples, timeouts, stats = generate_examples(args)
    log_dict.update(stats)
    split_and_write(examples, timeouts, args, log_dict)


if __name__ == '__main__':
    run()
