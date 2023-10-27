"""Functionalities for managing and calling configurators."""
from unified_planning.io import PDDLReader
from unified_planning.exceptions import UPProblemDefinitionError

from up_ac.AC_interface import *

import json
import timeit
import time


class Configurator():
    """Configurator functions."""

    def __init__(self):
        """Initialize generic interface."""
        self.capabilities = {'quality': {
                             'OneshotPlanner': 
                             ['lpg', 'fast-downward', 'enhsp', 'symk'],
                             'AnytimePlanner': ['fast-downward', 'symk']},
                             'runtime': {
                             'OneshotPlanner':
                             ['lpg', 'fast-downward', 'enhsp', 'symk',
                              'tamer', 'pyperplan', 'fmap'],
                             'AnytimePlanner': ['fast-downward', 'symk']}
                             }
        self.incumbent = None
        self.instance_features = {}
        self.train_set = {}
        self.test_set = {}
        self.reader = PDDLReader()
        self.metric = None
        self.crash_cost = 0
        self.ac = None
        self.verbose = True

    def print_feedback(self, engine, instance, feedback):
        """
        Print feedback from the engine.

        :param engine: Name of the engine.
        :type engine: str
        :param instance: Name of the instance.
        :type instance: str
        :param feedback: Feedback from the engine.
        """
        if self.verbose:
            print(f'** Feedback of {engine} on instance\n**' +
                  f' {instance}\n** is {feedback}\n\n')

    def get_instance_features(self, instance_features=None):
        """
        Save instance features.

        :param instance_features: Instance names and their features in lists.
        :type instance_features: dict, optional
        """
        self.instance_features = instance_features
        if self.verbose:
            print('\nSetting instance features.\n')

    def set_training_instance_set(self, train_set):
        """
        Save training instance set.

        :param train_set: List of instance paths.
        :type train_set: list
        """
        self.train_set = train_set
        if self.verbose:
            print('\nSetting training instance set.\n')

    def set_test_instance_set(self, test_set):
        """
        Set test instance set.

        :param test_set: List of instance paths.
        :type test_set: list
        """
        self.test_set = test_set
        if self.verbose:
            print('\nSetting testing instance set.\n')

    def get_feedback_function(self, gaci, engine, metric, mode,
                              gray_box=False):
        """
        Generate the function to run the engine and get feedback.

        :param gaci: Algorithm Configuration interface object.
        :type gaci: ACInterface
        :param engine: Engine name.
        :type engine: str
        :param metric: Metric, either 'runtime' or 'quality'.
        :type metric: str
        :param mode: Type of planning.
        :type mode: str
        :param gray_box: True if gray box to be used, optional.
        :type gray_box: bool, optional

        :return: Planner feedback function or None if not supported.
        :rtype: function or None
        """
        if engine in self.capabilities[metric][mode]:
            self.metric = metric

            planner_feedback = None

            return planner_feedback
        else:
            if self.verbose:
                print(f'Algorithm Configuration for {metric} of {engine}' + \
                      f' in {mode} is not supported.')
            return None

    def set_scenario(self, engine, param_space, gaci,
                     configuration_time=120, n_trials=400, min_budget=1,
                     max_budget=3, crash_cost=0, planner_timelimit=30,
                     n_workers=1, instances=[], instance_features=None,
                     metric='runtime', popSize=128, evalLimit=2147483647):
        """
        Set up algorithm configuration scenario.

        :param engine: Engine name.
        :type engine: str
        :param param_space: ConfigSpace object.
        :type param_space: ConfigSpace
        :param gaci: AC interface object.
        :type gaci: ACInterface
        :param configuration_time: Overall configuration time budget, optional.
        :type configuration_time: int, optional
        :param n_trials: Maximum number of engine evaluations, optional.
        :type n_trials: int, optional
        :param min_budget: Minimum number of instances to use, optional.
        :type min_budget: int, optional
        :param max_budget: Maximum number of instances to use, optional.
        :type max_budget: int, optional
        :param crash_cost: Cost to use if the engine fails, optional.
        :type crash_cost: int, optional
        :param planner_timelimit: Maximum runtime per evaluation, optional.
        :type planner_timelimit: int, optional
        :param n_workers: Number of cores to utilize, optional.
        :type n_workers: int, optional
        :param instances: Problem instance paths, optional.
        :type instances: list, optional
        :param instance_features: Instance names and lists of features, optional.
        :type instance_features: dict, optional
        :param metric: Optimization metric, optional.
        :type metric: str, optional
        :param popSize: Population size of configs per generation (OAT), optional.
        :type popSize: int, optional
        :param evalLimit: Maximum number of evaluations (OAT), optional.
        :type evalLimit: int, optional
        """

        scenario = None

        self.scenario = scenario

    def optimize(self, feedback_function=None, gray_box=False):
        """
        Run the algorithm configuration.

        :param feedback_function: Function to run engine and get feedback, optional.
        :type feedback_function: function, optional
        :param gray_box: True if gray box usage, optional.
        :type gray_box: bool, optional

        :return: The best configuration found during optimization.
        :rtype: dict
        """
        if feedback_function is not None:

            return self.incumbent

    def evaluate(self, metric, engine, mode, incumbent, gaci,
                 planner_timelimit=10, crash_cost=0, instances=[]):
        """
        Evaluate performance of found configuration on training set.

        :param metric: Optimization metric.
        :type metric: str
        :param engine: Engine name.
        :type engine: str
        :param mode: Planning mode.
        :type mode: str
        :param incumbent: Parameter configuration to evaluate.
        :type incumbent: dict
        :param gaci: AC interface object.
        :type gaci: ACInterface
        :param planner_timelimit: Max runtime per evaluation, optional.
        :type planner_timelimit: int, optional
        :param crash_cost: Cost if engine fails, optional.
        :type crash_cost: int, optional
        :param instances: Instance paths, optional.
        :type instances: list, optional

        :return: Average performance on the instances.
        :rtype: float
        """
        if incumbent is not None:
            # Avoid (meaningless) pebble AssertionError on Google Colab
            try:
                import google.colab
                IN_COLAB = True
            except:
                IN_COLAB = False

            if not instances:
                instances = self.test_set
            nr_inst = len(instances)
            avg_f = 0
            for inst in instances:
                if metric == 'runtime':
                    from pebble import concurrent
                    from concurrent.futures import TimeoutError
                    start = timeit.default_timer()

                instance_p = f'{inst}'
                domain_path = instance_p.rsplit('/', 1)[0]
                domain = f'{domain_path}/domain.pddl'
                pddl_problem = self.reader.parse_problem(f'{domain}',
                                                         f'{instance_p}')

                try:
                    if metric == 'runtime':
                        if IN_COLAB:
                            def solve(incumbent, metric, engine,
                                      mode, pddl_problem):
                                f = \
                                    gaci.run_engine_config(incumbent,
                                                           metric, engine,
                                                           mode, pddl_problem)

                                return f
                        else:
                            @concurrent.process(timeout=planner_timelimit)
                            def solve(incumbent, metric, engine,
                                      mode, pddl_problem):
                                f = \
                                    gaci.run_engine_config(incumbent,
                                                           metric, engine,
                                                           mode, pddl_problem)

                                return f

                        f = solve(incumbent, metric, engine,
                                  mode, pddl_problem)
                        
                        try:
                            if IN_COLAB:
                                f = f
                            else:
                                f = f.result()
                        except (TimeoutError, AssertionError):
                            f = planner_timelimit
                    elif metric == 'quality':                    
                        f = \
                            gaci.run_engine_config(incumbent,
                                                   metric, engine,
                                                   mode, pddl_problem)

                except (AssertionError, NotImplementedError,
                        UPProblemDefinitionError):
                    if self.verbose:
                        print('\n** Error in planning engine!')
                    if metric == 'runtime':
                        f = planner_timelimit
                    elif metric == 'quality':
                        f = crash_cost

                if metric == 'runtime':
                    if f is None:
                        f = planner_timelimit
                    elif 'measure':
                        f = timeit.default_timer() - start
                        if f > planner_timelimit:
                            f = planner_timelimit

                if f is not None and self.metric == 'quality':
                    f = -f
                if f is not None: 
                    avg_f += f
                else:
                    avg_f += self.crash_cost
                if metric == 'runtime':
                    if self.verbose:
                        print(f'\nFeedback on instance {inst}:\n\n', f, '\n')
                elif metric == 'quality':
                    if f is not None:
                        if self.verbose:
                            print(f'\nFeedback on instance {inst}:\n\n', -f,
                                  '\n')
                    else:
                        if self.verbose:
                            print(f'\nFeedback on instance {inst}:\n\n', None,
                                  '\n')
            if nr_inst != 0:
                avg_f = avg_f / nr_inst
                if metric == 'runtime':
                    print(f'\nAverage performance on {nr_inst} instances:',
                          avg_f, 'seconds\n')
                if metric == 'quality':
                    print(f'\nAverage quality performance on {nr_inst} instances:',
                          -avg_f, '\n')
                return avg_f
            else:
                print('\nPerformance could not be evaluated. No plans found.')
                return None
        else:
            return None

    def save_config(self, path, config, gaci, engine):
        """
        Save configuration in json file.

        :param path: Path where to save.
        :type path: str
        :param config: Configuration to save.
        :type config: dict
        :param gaci: AC interface object.
        :type gaci: ACInterface
        :param engine: Engine name.
        :type engine: str
        """
        if config is not None:
            config = gaci.transform_conf_from_ac(engine, config)
            with open(f'{path}/incumbent_{engine}.json', 'w') as f:
                json.dump(config, f)
            if self.verbose:
                print('\nSaved best configuration in ' +
                      f'{path}/incumbent_{engine}.json\n')
        else:
            if self.verbose:
                print(f'No configuration was saved. It was {config}')
