"""Functionalities for managing and calling SMAC."""
from smac import Scenario
from smac import AlgorithmConfigurationFacade
from unified_planning.exceptions import UPProblemDefinitionError, UPException
import os
import dill
import sys

import timeit
import signal
from contextlib import contextmanager
from pebble import concurrent
from pebble.common import ProcessExpired

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from AC_interface import *
from configurators import Configurator
from utils.patch_daskrunner import patch_daskrunner

# We do this only to be able to set the SMAC output dir!
AlgorithmConfigurationFacade = patch_daskrunner(AlgorithmConfigurationFacade)


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


class SmacConfigurator(Configurator):
    """Configurator functions."""

    def __init__(self):
        """Initialize Smac configurator."""
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

            def planner_feedback(config, instance, seed, reader):
                start = timeit.default_timer()
                instance_p = f'{instance}'
                # Since Smac handles time limits itself,
                # we do not use concurrent, as with other AC tools
                if metric == 'quality':
                    timelimit = self.planner_timelimit - self.patience
                else:
                    timelimit = self.planner_timelimit

                try:
                    if isinstance(self.train_set, dict) and \
                            isinstance(self.train_set[instance_p], tuple):
                        domain = self.train_set[instance_p][1]
                        problem = self.train_set[instance_p][0]
                        pddl_problem = reader.parse_problem(domain, problem)
                    else:
                        domain_path = instance_p.rsplit('/', 1)[0]
                        domain = f'{domain_path}/domain.pddl'
                        pddl_problem = reader.parse_problem(f'{domain}',
                                                            f'{instance_p}')

                    if engine == 'tamer' or engine == 'pyperplan':
                        @concurrent.process(timeout=timelimit)
                        def solve(config, metric, engine,
                                  mode, pddl_problem):
                            feedback = \
                                gaci.run_engine_config(config,
                                                       metric, engine,
                                                       mode, pddl_problem,
                                                       timelimit)

                            return feedback
                    else:
                        def solve(config, metric, engine,
                                  mode, pddl_problem):
                            feedback = \
                                gaci.run_engine_config(config,
                                                       metric, engine,
                                                       mode, pddl_problem,
                                                       timelimit)

                            return feedback

                    try:
                        if engine in ('tamer', 'pyperplan'):
                            future = solve(config, metric, engine,
                                           mode, pddl_problem)
                            try:
                                feedback = future.result()
                            except (TimeoutError, ProcessExpired) as err:
                                print(err)
                                if metric == 'runtime':
                                    feedback = timelimit
                                elif metric == 'quality':
                                    feedback = self.crash_cost

                        else:
                            with time_limit(timelimit):
                                feedback = solve(config, metric, engine,
                                                 mode, pddl_problem)

                    except TimeoutException:
                        if metric == 'runtime':
                            feedback = timelimit
                        elif metric == 'quality':
                            feedback = self.crash_cost

                except (AssertionError, NotImplementedError,
                        UPProblemDefinitionError, UPException,
                        UnicodeDecodeError) as err:
                    if self.verbose:
                        print('\n** Error in planning engine!', err)
                    if metric == 'runtime':
                        feedback = timelimit
                    elif metric == 'quality':
                        feedback = self.crash_cost

                if feedback == 'unsolvable':
                    if metric == 'runtime':
                        feedback = timelimit
                    elif metric == 'quality':
                        feedback = self.crash_cost
              
                if feedback is not None:
                    # SMAC always minimizes
                    if metric == 'quality':
                        self.print_feedback(engine, instance, feedback)
                        return feedback
                    # Solving runtime optimization by passing
                    # runtime as result, since smac minimizes it
                    elif metric == 'runtime':
                        if engine in ('tamer', 'pyperplan'):
                            feedback = timeit.default_timer() - start
                            if feedback > timelimit:
                                feedback = timelimit
                            self.print_feedback(engine, instance, feedback)
                        else:
                            if feedback > timelimit:
                                feedback = timelimit
                            self.print_feedback(engine, instance, feedback)
                        return feedback
                else:
                    # Penalizing failed runs
                    if metric == 'runtime':
                        # Penalty is max runtime in runtime scenario
                        feedback = timelimit
                        self.print_feedback(engine, instance, feedback)
                    elif metric == 'quality':
                        # Penalty is defined by user in quality scenario
                        feedback = self.crash_cost
                        self.print_feedback(engine, instance, feedback)

                    return feedback

            try:
                import up_ac
                path = '/' + os.path.abspath(up_ac.__file__).strip('/__init__.py')
                path += '/utils'
            except ImportError:
                path = os.getcwd().rsplit('up_ac', 1)[0]
                if path[-1] != "/":
                    path += "/"
                path += 'up_ac/utils'

            self.feedback_path = path

            dill.dump(
                planner_feedback, open(
                    f'{path}/feedback.pkl', 'wb'),
                recurse=True)

            return planner_feedback
        else:
            if self.verbose:
                print(f'Algorithm Configuration for {metric} of {engine}' + \
                      f' in {mode} is not supported.')
            return None

    def set_scenario(self, engine, param_space, gaci,
                     configuration_time=120, n_trials=400, min_budget=1,
                     max_budget=3, crash_cost=10000, planner_timelimit=30,
                     n_workers=1, instances=[], instance_features=None,
                     output_dir='smac3_output', metric='runtime',
                     patience=10):
        """
        Set up the algorithm configuration scenario for SMAC (Sequential Model-based Algorithm Configuration).

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
        :param patience: Extra time in seconds to grant engine to terminate gracefully.
        :type patience: int

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
        self.scenarios = {}

        for n in range(self.n_workers):
            scenario_dict = {'configspace': param_space,
                             'walltime_limit': configuration_time,
                             'n_trials': n_trials,
                             'min_budget': min_budget,
                             'max_budget': max_budget,
                             'deterministic': True,
                             'crash_cost': crash_cost,  
                             'trial_walltime_limit': planner_timelimit,  
                             'use_default_config': True,
                             'n_workers': 1,
                             'instances': instances,
                             'instance_features': instance_features}

            od = output_dir + f'_{n}'

            self.scenarios[n] = Scenario(**scenario_dict,
                                         output_directory=od)

        if self.verbose:
            print('\nSMAC scenario is set.\n')

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
            try:
                import up_ac
                path = \
                    '/' + os.path.abspath(up_ac.__file__).strip('/__init__.py')
                path += '/utils'
            except ImportError:
                path = os.getcwd().rsplit('up_ac', 1)[0]
                if path[-1] != "/":
                    path += "/"
                path += 'up_ac/utils'
            sys.path.append(r"{}".format(path))
            from load_smac_feedback import get_feedback

            if self.verbose:
                print('\nStarting Parameter optimization\n')

            ac_dict = {}
            self.incumbent_dict = {}
            self.incumbent_dicts_dict = {}
            self.runhistories = {}
            self.costs = []
            threads = []
            results = [None for n in range(self.n_workers)]

            def optimize(smac, results, n):
                results[n] = smac.optimize()

            import threading

            for n in range(self.n_workers):

                print('\n\n\n')
                print('!!!!!!!!!!!!!!')
                print('Running AC Nr.', n)
                print('!!!!!!!!!!!!!!')
                print('\n\n\n')
 
                ac_dict[n] = AlgorithmConfigurationFacade(
                    self.scenarios[n],
                    get_feedback,
                    overwrite=True)

                threads.append(threading.Thread(target=optimize,
                                                args=(ac_dict[n], results, n)))

            for n in range(self.n_workers):
                threads[n].start()

            for n in range(self.n_workers):
                threads[n].join()

            for n in range(self.n_workers):
                self.incumbent_dict[n] = results[n]
                if results[n] is not None:
                    self.incumbent_dicts_dict[n] = results[n].get_dictionary()
                    self.runhistories[n] = ac_dict[n]._runhistory
                    self.costs.append(
                        self.runhistories[n].get_cost(self.incumbent_dict[n]))
                else:
                    self.incumbent_dicts_dict[n] = None
                    self.runhistories[n] = None
                    if self.metric == 'quality':
                        self.costs.append(self.crash_cost)
                    else:
                        self.costs.append(self.planner_timelimit)

            for n, incumbent in self.incumbent_dicts_dict.items():
                print('Incumbent', n, incumbent, 'has cost', self.costs[n])

            index_best = \
                min(range(len(self.costs)), key=self.costs.__getitem__)
            self.incumbent = self.incumbent_dicts_dict[index_best]

            if self.verbose:
                print('\nBest Configuration found is:\n',
                      self.incumbent)

            return self.incumbent, None
        else:
            return None, None
