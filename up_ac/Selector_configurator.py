"""Functionalities for managing and calling selector."""
import os
import sys
import shutil
import json
from ConfigSpace.read_and_write import pcs

from selector.run_ac import ac

from AC_interface import *
from configurators import Configurator

import signal
from contextlib import contextmanager

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class SelectorConfigurator(Configurator):
    """Configurator functions."""

    def __init__(self):
        """Initialize Selector configurator."""
        Configurator.__init__(self)
        self.crash_cost = 0
        self.planner_timelimit = 0
        self.engine = None
        self.gaci = None 

    def get_feedback_function(self, gaci, engine, metric, mode,
                              gray_box=False):
        """
        Generate a function to run the planning engine and obtain feedback.

        :param ACInterface gaci: Algorithm Configuration Interface object.
        :param str engine: Name of the planning engine.
        :param str metric: Metric to optimize, either 'runtime' or 'quality'.
        :param str mode: Type of planning.
        :param bool gray_box: True if using a gray box, False otherwise (optional).

        :return: A planner feedback function that takes configuration, instance, seed, and reader.
        :rtype: function

        :raises ValueError: If the provided engine is not supported for the given metric and mode.
        """

        if engine in self.capabilities[metric][mode]:
            self.metric = metric

            feedback_args = {'engine': engine, 'metric': metric,
                             'mode': mode, 'gray_box': gray_box,
                             'patience': self.patience,
                             'train_set': self.train_set,
                             'timelimit': self.scenario['cutoff_time'],
                             'crash_cost': self.crash_cost,
                             'verbose': self.verbose}

            fs_path = f'{self.feedback_path}/sel_feedback_setting'

            if not os.path.exists(fs_path):
                os.makedirs(fs_path)

            with open(f'{fs_path}/feedback_args.txt', 'w') as f:
                f.write(str(feedback_args))

            planner_feedback = f'{fs_path}/feedback_args.txt'

            return planner_feedback
        else:
            if self.verbose:
                print(f'''Algorithm Configuration for {metric} of {engine}
                      in {mode} is not supported.''')
            return None

    def set_scenario(self, engine, param_space, gaci,
                     configuration_time=120, n_trials=400, min_budget=1,
                     max_budget=3, crash_cost=0, planner_timelimit=30,
                     n_workers=1, instances=[], instance_features=None,
                     tourn_size=8, metric='runtime', memory_limit=2048,
                     patience=10, output_dir='up_ac_selector',
                     ray_mode='desktop'):
        """
        Set up the algorithm configuration scenario for Selector.

        :param str engine: The name of the planning engine.
        :param ConfigSpace.ConfigurationSpace param_space: ConfigSpace object defining the parameter space.
        :param ACInterface gaci: AC interface object.
        :param int configuration_time: Overall configuration time budget in seconds (default is 120).
        :param int n_trials: Maximum number of engine evaluations (default is 400).
        :param int min_budget: Minimum number of instances to use (default is 1).
        :param int max_budget: Maximum number of instances to use (default is 3).
        :param int crash_cost: The cost to use if the engine fails (default is 0).
        :param int planner_timelimit: Maximum runtime per evaluation for the planner (default is 30).
        :param int n_workers: Number of cores to utilize (default is 1).
        :param list instances: List of problem instance paths (default is empty list, uses train_set).
        :param dict instance_features: Dictionary containing instance names and lists of features (default is None).
        :param str metric: The optimization metric, either 'runtime' or 'quality' (default is 'runtime').
        :param int memory_limit: At which amount of MB used by a planning engine to terminate the run.
        :param int patience: Extra time in seconds to grant engine to terminate gracefully.
        :param str output_dir: Name of the directory selector writes to (directory is at .)
        :param str ray_mode: Set to 'desktop' if running locally, or 'cluster' if running in SLURM

        :raises ValueError: If an unsupported metric is provided.
        """
        if not instances:
            instances = self.train_set
        if isinstance(instances[0], tuple):
            train_set_dict = {}
            clean_inst = instances
            instances = []
            for inst in clean_inst:
                train_set_dict[inst[0]] = inst
                instances.append(inst[0])
            self.train_set = train_set_dict
        path = os.getcwd().rsplit('up_ac', 1)[0]
        if path[-1] != "/":
            path += "/"
        path += 'up_ac/utils'

        self.feedback_path = path
        with open(f'{self.feedback_path}/instances.txt', 'w') as f:
            for inst in instances:
                f.write("%s\n" % inst)
        with open(f'{self.feedback_path}/features.txt', 'w') as f:
            header = ['inst_name'] + [str(feat_nr) 
                                      for feat_nr in range(
                                          len(instance_features[list(
                                              instance_features.keys())[0]]))]
            f.write("%s\n" % ', '.join(header))
            for inst, feats in instance_features.items():
                feats = map(str, feats)
                f.write("%s, %s\n" % (inst, ', '.join(feats)))
        with open(f'{self.feedback_path}/configspace.pcs', 'w') as f:
            f.write(pcs.write(param_space))
        if metric == 'runtime':
            if crash_cost != planner_timelimit:
                crash_cost = planner_timelimit
        self.crash_cost = crash_cost
        self.patience = patience
        self.metric = metric
        if self.metric == 'quality':
            self.planner_timelimit = planner_timelimit + patience
        else:
            self.planner_timelimit = planner_timelimit
        self.engine = engine
        self.gaci = gaci
        self.n_workers = n_workers
        number_tournaments = int(n_workers / tourn_size)
        path = os.getcwd().rsplit('up_ac', 1)[0]
        if path[-1] != "/":
            path += "/"
        path += 'up_ac/utils'
        shutil.copyfile(f'{path}/up_ac_wrapper.py', f'up_ac_wrapper.py')
        if len(self.train_set) < 8:
            smac_pca_dim = len(self.train_set)
        else:
            smac_pca_dim = 8

        scen_files = {'paramfile': f'{self.feedback_path}/configspace.pcs',
                      'instance_file': f'{self.feedback_path}/instances.txt',
                      'feature_file': f'{self.feedback_path}/features.txt'}

        self.scenario = \
            dict(scen_files=scen_files, ray_mode=ray_mode,
                 run_obj=metric, overall_obj='PAR10',
                 cutoff_time=planner_timelimit, seed=44, par=10,
                 winners_per_tournament=1, tournament_size=tourn_size,
                 number_tournaments=number_tournaments,
                 termination_criterion='total_runtime',
                 monitor='tournament_level',
                 initial_instance_set_size=min_budget,
                 set_size=max_budget, generator_multiple=1,
                 memory_limit=memory_limit, 
                 check_path=False, log_folder=output_dir,
                 wallclock_limit=configuration_time,
                 wrapper_mod_name='up_ac_wrapper', deterministic=0,
                 cleanup=True, wrapper_class_name='UP_AC_Wrapper',
                 smac_pca_dim=smac_pca_dim, crash_cost=crash_cost,
                 cpu_binding=True)

        if metric == 'runtime':
            self.scenario['solve_match'] = r'Tuned plan runtime is:'
            self.scenario['runtime_feedback'] = \
                r"\s*([+-]?\d+(?:\.\d+)?)"

        elif metric == 'quality':
            self.scenario['quality_match'] = r'Tuned plan quality is:'
            self.scenario['quality_extract'] = r'[-+]?\d+'

        if self.verbose:
            print('\nSelector scenario is set.\n')

    def optimize(self, feedback_function=None, gray_box=False):
        """
        Run the algorithm configuration optimization.

        :param function feedback_function: A function to run the engine and obtain feedback (optional).
        :param bool gray_box: True if gray box usage is enabled, False otherwise (optional).

        :return: A tuple containing the best configuration found and additional information (if available).
                Returns None if feedback_function is not provided.
        :rtype: tuple or None
        """
        if feedback_function is not None:
            # Import feedback function, since dask cannot pickle local objects            
            path = os.getcwd().rsplit('up_ac', 1)[0]
            if path[-1] != "/":
                path += "/"
            path += 'up_ac/utils'
            sys.path.append(r"{}".format(path))

            if self.verbose:
                print('\nStarting Parameter optimization\n')

            __name__ = '__main__'

            ac(self.scenario.pop('scen_files'), self.scenario.pop('ray_mode'),
               **self.scenario)

            os.remove('up_ac_wrapper.py')

            lf = self.scenario['log_folder']

            with open(f'./selector/logs/{lf}/overall_best.json') as f:
                self.incumbent = json.load(f)

            self.incumbent = \
                self.incumbent[list(self.incumbent.keys())[0]]['conf']

            if self.verbose:
                print('\nBest Configuration found is:\n',
                      self.incumbent)

            return self.incumbent, None
        else:
            return None, None