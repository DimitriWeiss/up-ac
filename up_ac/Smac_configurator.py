"""Functionalities for managing and calling configurators."""
from smac import Scenario
from smac import AlgorithmConfigurationFacade
from unified_planning.exceptions import UPProblemDefinitionError
from pebble import concurrent
from concurrent.futures import TimeoutError

from up_ac.AC_interface import *
from up_ac.configurators import Configurator

import timeit


class SmacConfigurator(Configurator):
    """Configurator functions."""

    def __init__(self):
        """Initialize Smac configurator."""
        Configurator.__init__(self)

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
        print(metric)
        print(mode)
        print(self.capabilities)
        if engine in self.capabilities[metric][mode]:
            self.metric = metric

            def planner_feedback(config, instance, seed=0):
                start = timeit.default_timer()
                print(instance)
                instance_p = f'{instance}'
                domain_path = instance_p.rsplit('/', 1)[0]
                domain = f'{domain_path}/domain.pddl'
                pddl_problem = self.reader.parse_problem(f'{domain}',
                                                         f'{instance_p}')

                '''
                feedback = \
                    gaci.run_engine_config(config,
                                           metric,
                                           engine,
                                           mode,
                                           pddl_problem)
                '''

                try:
                    @concurrent.process(timeout=self.planner_timelimit)
                    def solve(config, metric, engine,
                              mode, pddl_problem):
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

            return planner_feedback
        else:
            print(f'Algorithm Configuration for {metric} of {engine}' + \
                  f' in {mode} is not supported.')
            return None

    def set_scenario(self, ac_tool, engine, param_space, gaci,
                     configuration_time=120, n_trials=400, min_budget=1,
                     max_budget=3, crash_cost=0, planner_timelimit=30,
                     n_workers=1, instances=[], instance_features=None,
                     metric='runtime'):
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
        print('\nSMAC scenario is set.\n')

        self.scenario = scenario

    def optimize(self, ac_tool, feedback_function=None, gray_box=False):
        """
        Run the algorithm configuration.

        parameter ac_tool: str, which AC tool.
        parameter feedback_function: function to run engine and get feedback.
        parameter gray_box: True, if gray box usage.
        """
        if feedback_function is not None:
            print('\nStarting Parameter optimization\n')
            ac = AlgorithmConfigurationFacade(
                self.scenario,
                feedback_function,
                overwrite=True)

            self.incumbent = ac.optimize()

            self.incumbent = self.incumbent.get_dictionary()

            print(f'\nBest Configuration found by {ac_tool} is:\n',
                  self.incumbent)

            return self.incumbent, None
        else:
            return None, None
