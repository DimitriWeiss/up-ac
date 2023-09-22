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
        Print feedback from engine.

        parameter engine: str, which engine.
        parameter instance: str, which instance.
        parameter feedback: feedback from engine.
        """
        print(f'** Feedback of {engine} on instance\n**' +
              f' {instance}\n** is {feedback}\n\n')

    def get_instance_features(self, instance_features=None):
        """
        Save instance features.

        parameter instance_features: dict, inst name and features in lists.
        """
        self.instance_features = instance_features
        print('\nSetting instance features.\n')

    def set_training_instance_set(self, train_set):
        """
        Save training instance set.

        parameter train_set: list, instance paths.
        """
        self.train_set = train_set
        print('\nSetting training instance set.\n')

    def set_test_instance_set(self, test_set):
        """
        Save test instance set.

        parameter test_set: list, instance paths.
        """
        self.test_set = test_set
        print('\nSetting testing instance set.\n')

    def get_feedback_function(self, gaci, engine, metric, mode,
                              gray_box=False):
        """
        Generate the function to run engine and get feedback.

        parameter gaci: AC interface object.
        parameter engine: str, engine name.
        parameter metric: str, 'runtime' or 'quality'
        parameter mode: str, type of planning.
        parameter gray_box: True, if gra box to use

        return planner_feedback: function, planner feedback function.
        """
        if engine in self.capabilities[metric][mode]:
            self.metric = metric

            planner_feedback = None

            return planner_feedback
        else:
            print(f'Algorithm Configuration for {metric} of {engine}' + \
                  f' in {mode} is not supported.')
            return None

    def set_scenario(self, ac_tool, engine, param_space, gaci,
                     configuration_time=120, n_trials=400, min_budget=1,
                     max_budget=3, crash_cost=0, planner_timelimit=30,
                     n_workers=1, instances=[], instance_features=None,
                     metric='runtime', popSize=128, evlaLimit=2147483647):
        """
        Set up algorithm configuration scenario.

        parameter ac_tool: str, which configuration tol.
        parameter engine: str, which engine.
        parameter param_space: ConfigSpace object.
        parameter gaci: AC interface object.
        parameter configuration_time: int, overall configuration time budget.
        parameter n_trials: int, max number of engine evaluations.
        parameter min_budget: int, min number of instances to use.
        parameter max_budget: int, max number of instances to use.
        parameter crash_cost: int, which cost to use if engine fails.
        parameter planner_timelimit: int, max runtime per evaluation.
        parameter n_workers: int, no. of cores to utilize.
        parameter instances: list, problem instance paths.
        parameter instance_features: dict, inst names and lists of features.
        parameter metric: str, optimization metric.
        parameter popSize: int, populaton size of configs per generation (OAT).
        parameter evlaLimit: int, max no. of evaluations (OAT).
        """

        scenario = None

        self.scenario = scenario

    def optimize(self, ac_tool, feedback_function=None, gray_box=False):
        """
        Run the algorithm configuration.

        parameter ac_tool: str, which AC tool.
        parameter feedback_function: function to run engine and get feedback.
        parameter gray_box: True, if gray box usage.
        """
        if feedback_function is not None:

            return self.incumbent

    def evaluate(self, ac_tool, metric, engine, mode, incumbent, gaci,
                 planner_timelimit=10, crash_cost=0, instances=[]):
        """
        Evaluate performance of found configuration on training set.

        parameter ac_tool: str, which AC tool.
        parameter metric: str, which optimization metric.
        parameter engine: str, which engine.
        parameter mode: str, which Planning mode.
        parameter incumbent: dict, parameter configuration to evaluate.
        parameter gaci: AC interface object.
        parameter planner_timelimit: int, max runtime per evaluation.
        parameter crash_cost: int, cost if engine fails.
        parameter instances: list, optional, if test set not declared in gaci.
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
                        def solve(ac_tool, incumbent, metric, engine,
                                  mode, pddl_problem):
                            f = \
                                gaci.run_engine_config(incumbent,
                                                       metric, engine,
                                                       mode, pddl_problem)

                            return f

                        f = solve(ac_tool, incumbent, metric, engine,
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

        parameter path: str, path where to save.
        parameter config: dict, config to save.
        parameter gaci. AC interface object.
        parameter engine: str, which engine.
        """
        if config is not None:
            config = gaci.transform_conf_from_ac(engine, config)
            with open(f'{path}/incumbent_{engine}.json', 'w') as f:
                json.dump(config, f)
            print('\nSaved best configuration in ' +
                  f'{path}/incumbent_{engine}.json\n')
        else:
            print(f'No configuration was saved. It was {config}')
