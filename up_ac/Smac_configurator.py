"""Functionalities for managing and calling configurators."""
from smac import Scenario
from smac import AlgorithmConfigurationFacade
from unified_planning.exceptions import UPProblemDefinitionError, UPException
from pebble import concurrent
import os
import dill
import sys
from concurrent.futures import TimeoutError

from up_ac.AC_interface import *
from up_ac.configurators import Configurator

import timeit


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
                domain_path = instance_p.rsplit('/', 1)[0]
                domain = f'{domain_path}/domain.pddl'
                pddl_problem = reader.parse_problem(f'{domain}',
                                                    f'{instance_p}')

                # Since Smac handles time limits itself,
                # we do not use concurrent, as with other AC tools
                try:
                    feedback = \
                        gaci.run_engine_config(config,
                                               metric,
                                               engine,
                                               mode,
                                               pddl_problem)
                except (AssertionError, NotImplementedError,
                        UPProblemDefinitionError, UPException,
                        UnicodeDecodeError) as err:
                    if self.verbose:
                        print('\n** Error in planning engine!', err)
                    if metric == 'runtime':
                        feedback = self.planner_timelimit
                    elif metric == 'quality':
                        feedback = self.crash_cost
                '''

                try:
                    @concurrent.process(timeout=self.planner_timelimit)
                    def solve(config, metric, engine,
                              mode, pddl_problem):
                        print('Running further\n')
                        feedback = \
                            gaci.run_engine_config(config,
                                                   metric, engine,
                                                   mode, pddl_problem)

                        return feedback

                    feedback = solve(config, metric, engine,
                                     mode, pddl_problem)
                    
                    try:
                        feedback = feedback.result()
                    except TimeoutError:
                        if metric == 'runtime':
                            feedback = self.planner_timelimit
                        elif metric == 'quality':
                            feedback = self.crash_cost

                except (AssertionError, NotImplementedError,
                        UPProblemDefinitionError):
                    print('\n** Error in planning engine!')
                    if metric == 'runtime':
                        feedback = self.planner_timelimit
                    elif metric == 'quality':
                        feedback = self.crash_cost
                '''
              
                if feedback is not None:
                    # SMAC always minimizes
                    if metric == 'quality':
                        self.print_feedback(engine, instance, feedback)
                        return -feedback
                    # Solving runtime optimization by passing
                    # runtime as result, since smac minimizes it
                    elif metric == 'runtime':
                        if engine in ('tamer', 'pyperplan'):
                            feedback = timeit.default_timer() - start
                            self.print_feedback(engine, instance, feedback)
                        else:
                            feedback = feedback
                            self.print_feedback(engine, instance, feedback)
                        return feedback
                else:
                    # Penalizing failed runs
                    if metric == 'runtime':
                        # Penalty is max runtime in runtime scenario
                        feedback = self.scenario.trial_walltime_limit
                        self.print_feedback(engine, instance, feedback)
                    else:
                        # Penalty is defined by user in quality scenario
                        feedback = self.crash_cost
                        self.print_feedback(engine, instance, feedback)

                    return feedback

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
                     max_budget=3, crash_cost=0, planner_timelimit=30,
                     n_workers=1, instances=[], instance_features=None,
                     metric='runtime'):
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

        :raises ValueError: If an unsupported metric is provided.
        """
        if not instances:
            instances = self.train_set
        self.crash_cost = crash_cost
        self.planner_timelimit = planner_timelimit
        self.engine = engine
        self.gaci = gaci
        scenario = Scenario(
            param_space,
            # We want to optimize for <configuration_time> seconds
            walltime_limit=configuration_time,
            n_trials=n_trials,  # Maximum number or algorithm runs
            min_budget=min_budget,  # Use min <min_budget> instance
            max_budget=max_budget,  # Use max <max_budget> instances
            deterministic=True,  # Not stochastic algorithm
            # Cost of algorithm crashing -> AC metric to evaluate configuration
            crash_cost=crash_cost,  
            # Max time for algorithm to run
            trial_walltime_limit=planner_timelimit,  
            use_default_config=True,  # include default config
            n_workers=n_workers,  # Number of parallel runs
            instances=instances,  # List of training instances
            instance_features=instance_features  # Dict of instance features
        )
        if self.verbose:
            print('\nSMAC scenario is set.\n')

        self.scenario = scenario

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
            from load_smac_feedback import get_feedback

            if self.verbose:
                print('\nStarting Parameter optimization\n')
 
            ac = AlgorithmConfigurationFacade(
                self.scenario,
                get_feedback,
                overwrite=True)

            self.incumbent = ac.optimize()

            self.incumbent = self.incumbent.get_dictionary()

            if self.verbose:
                print('\nBest Configuration found is:\n',
                      self.incumbent)

            return self.incumbent, None
        else:
            return None, None
