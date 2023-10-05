"""Functionalities for managing and calling configurators."""
from unified_planning.io import PDDLReader
from unified_planning.exceptions import UPProblemDefinitionError

from up_ac.AC_interface import *

import json
import timeit


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

    def print_feedback(self, engine, instance, feedback):
        """
        Print feedback from the engine.

        Parameters:
            engine (str): Name of the engine.
            instance (str): Name of the instance.
            feedback: Feedback from the engine.
        """
        print(f'** Feedback of {engine} on instance\n**' +
              f' {instance}\n** is {feedback}\n\n')

    def get_instance_features(self, instance_features=None):
        """
        Save instance features.

        Parameters:
            instance_features (dict): Instance names and their features in lists.
        """
        self.instance_features = instance_features
        print('\nSetting instance features.\n')

    def set_training_instance_set(self, train_set):
        """
        Save training instance set.

        Parameters:
            train_set (list): List of instance paths.
        """
        self.train_set = train_set
        print('\nSetting training instance set.\n')

    def set_test_instance_set(self, test_set):
        """
        Save test instance set.

        Parameters:
            test_set (list): List of instance paths.
        """
        self.test_set = test_set
        print('\nSetting testing instance set.\n')

    def get_feedback_function(self, gaci, engine, metric, mode,
                              gray_box=False):
        """
        Generate the function to run the engine and get feedback.

        Parameters:
            gaci (ACInterface): Algorithm Configuration interface object.
            engine (str): Engine name.
            metric (str): Metric, either 'runtime' or 'quality'.
            mode (str): Type of planning.
            gray_box (bool, optional): True if gray box to be used.

        Returns:
            function or None: Planner feedback function or None if not supported.
        """
        if engine in self.capabilities[metric][mode]:
            self.metric = metric

            planner_feedback = None

            return planner_feedback
        else:
            print(f'Algorithm Configuration for {metric} of {engine}' + \
                  f' in {mode} is not supported.')
            return None

    def set_scenario(self, engine, param_space, gaci,
                     configuration_time=120, n_trials=400, min_budget=1,
                     max_budget=3, crash_cost=0, planner_timelimit=30,
                     n_workers=1, instances=[], instance_features=None,
                     metric='runtime', popSize=128, evlaLimit=2147483647):
        """
        Set up algorithm configuration scenario.

        Parameters:
            engine (str): Engine name.
            param_space (ConfigSpace): ConfigSpace object.
            gaci (ACInterface): AC interface object.
            configuration_time (int, optional): Overall configuration time budget.
            n_trials (int, optional): Maximum number of engine evaluations.
            min_budget (int, optional): Minimum number of instances to use.
            max_budget (int, optional): Maximum number of instances to use.
            crash_cost (int, optional): Cost to use if the engine fails.
            planner_timelimit (int, optional): Maximum runtime per evaluation.
            n_workers (int, optional): Number of cores to utilize.
            instances (list, optional): Problem instance paths.
            instance_features (dict, optional): Instance names and lists of features.
            metric (str, optional): Optimization metric.
            popSize (int, optional): Population size of configs per generation (OAT).
            evlaLimit (int, optional): Maximum number of evaluations (OAT).

        """

        scenario = None

        self.scenario = scenario

    def optimize(self, feedback_function=None, gray_box=False):
        """
        Run the algorithm configuration.

        Parameters:
            feedback_function (function, optional): Function to run engine and get feedback.
            gray_box (bool, optional): True if gray box usage.

        Returns:
            incumbent: The best configuration found during optimization.
        """
        if feedback_function is not None:

            return self.incumbent

    def evaluate(self, metric, engine, mode, incumbent, gaci,
                 planner_timelimit=10, crash_cost=0, instances=[]):
        """
        Evaluate performance of found configuration on training set.

        Parameters:
            metric (str): Optimization metric.
            engine (str): Engine name.
            mode (str): Planning mode.
            incumbent (dict): Parameter configuration to evaluate.
            gaci: AC interface object.
            planner_timelimit (int, optional): Max runtime per evaluation.
            crash_cost (int, optional): Cost if engine fails.
            instances (list, optional): Instance paths.

        Returns:
            float: Average performance on the instances.
        """
        if incumbent is not None:
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
                            f = f.result()
                        except TimeoutError:
                            f = planner_timelimit
                    elif metric == 'quality':                    
                        f = \
                            gaci.run_engine_config(incumbent,
                                                   metric, engine,
                                                   mode, pddl_problem)

                except (AssertionError, NotImplementedError,
                        UPProblemDefinitionError):
                    print('\n** Error in planning engine!')
                    if metric == 'runtime':
                        f = planner_timelimit
                    elif metric == 'quality':
                        f = crash_cost

                if metric == 'runtime':
                    if f is None:
                        f = planner_timelimit
                    elif f == 'measure':
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
                    print(f'\nFeedback on instance {inst}:\n\n', f, '\n')
                elif metric == 'quality':
                    if f is not None:
                        print(f'\nFeedback on instance {inst}:\n\n', -f, '\n')
                    else:
                        print(f'\nFeedback on instance {inst}:\n\n', None,
                              '\n')
            if nr_inst != 0:
                avg_f = avg_f / nr_inst
                if metric == 'runtime':
                    print(f'\nAverage performance on {nr_inst} instances:',
                          avg_f, '\n')
                if metric == 'quality':
                    print(f'\nAverage performance on {nr_inst} instances:',
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

        Parameters:
            path (str): Path where to save.
            config (dict): Configuration to save.
            gaci: AC interface object.
            engine (str): Engine name.
        """
        if config is not None:
            config = gaci.transform_conf_from_ac(engine, config)
            with open(f'{path}/incumbent_{engine}.json', 'w') as f:
                json.dump(config, f)
            print('\nSaved best configuration in ' +
                  f'{path}/incumbent_{engine}.json\n')
        else:
            print(f'No configuration was saved. It was {config}')
