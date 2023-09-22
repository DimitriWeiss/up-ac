"""Functionalities for managing and calling configurators."""
from irace import irace
from unified_planning.exceptions import UPProblemDefinitionError
from pebble import concurrent
from concurrent.futures import TimeoutError

from up_ac.AC_interface import *
from up_ac.configurators import Configurator

import timeit


class IraceConfigurator(Configurator):
    """Configurator functions."""

    def __init__(self):
        """Initialize Irace configurator."""
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
        if engine in self.capabilities[metric][mode]:
            self.metric = metric

            def planner_feedback(experiment, scenario):
                start = timeit.default_timer()
                instance_p = \
                    self.scenario['instances'][experiment['id.instance'] - 1]
                domain_path = instance_p.rsplit('/', 1)[0]
                domain = f'{domain_path}/domain.pddl'
                pddl_problem = self.reader.parse_problem(f'{domain}',
                                                         f'{instance_p}')
                config = dict(experiment['configuration'])

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
                    if metric == 'quality':
                        self.print_feedback(engine, instance_p, feedback)
                        runtime = timeit.default_timer() - start
                        feedback = {'cost': -feedback, 'time': runtime}
                        return feedback
                    elif metric == 'runtime':
                        if engine in ('tamer', 'pyperplan'):
                            feedback = timeit.default_timer() - start
                            self.print_feedback(engine, instance_p, feedback)
                            feedback = {'cost': feedback, 'time': feedback}
                        else:
                            feedback = feedback
                            self.print_feedback(engine, instance_p, feedback)
                            feedback = {'cost': feedback, 'time': feedback}
                        return feedback
                else:
                    # Penalizing failed runs
                    if metric == 'runtime':
                        # Penalty is max runtime in runtime scenario
                        feedback = self.scenario['boundMax']
                        self.print_feedback(engine, instance_p, feedback)
                        feedback = {'cost': feedback, 'time': feedback}
                    else:
                        # Penalty is defined by user in quality scenario
                        feedback = self.crash_cost
                        self.print_feedback(engine, instance_p, feedback)
                        feedback = {'cost': feedback, 'time': feedback}

                    return feedback

            return planner_feedback
        else:
            print(f'Algorithm Configuration for {metric} of {engine} in' + 
                  f' {mode} is not supported.')
            return None

    def set_scenario(self, ac_tool, engine, param_space, gaci,
                     configuration_time=120,
                     n_trials=400, min_budget=1, max_budget=3, crash_cost=0,
                     planner_timelimit=30, n_workers=1, instances=[],
                     instance_features=None, metric='runtime'):
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
        default_conf, forbiddens = gaci.get_ps_irace(param_space)

        if metric == 'quality':
            test_type = 'friedman'
            capping = False
        elif metric == 'runtime':
            test_type = 't-test'
            capping = True
        # https://mlopez-ibanez.github.io/irace/reference/defaultScenario.html
        if forbiddens:
            scenario = dict(
                # We want to optimize for <configuration_time> seconds
                maxTime=configuration_time,  
                instances=instances,
                # List of training instances
                debugLevel=3, 
                # number of decimal places to be considered for real parameters
                digits=10, 
                # Number of parallel runs
                parallel=n_workers, 
                forbiddenFile="forbidden.txt",
                logFile="",
                initConfigurations=default_conf,
                nbConfigurations=8,
                deterministic=True,
                testType=test_type,
                capping=capping,
                boundMax=planner_timelimit,
                firstTest=min_budget
            )
        else:
            scenario = dict(
                # We want to optimize for <configuration_time> seconds
                maxTime=configuration_time, 
                # List of training instances
                instances=instances, 
                debugLevel=3, 
                # number of decimal places to be considered for real parameters
                digits=10, 
                # Number of parallel runs
                parallel=n_workers, 
                logFile="",
                initConfigurations=default_conf,
                nbConfigurations=8,
                deterministic=True,
                testType=test_type,
                capping=capping,
                boundMax=planner_timelimit,
                firstTest=min_budget
            )

        self.irace_param_space = gaci.irace_param_space

        print('\nIrace scenario is set.\n')

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
            ac = irace(self.scenario,
                       self.irace_param_space,
                       feedback_function)
            self.incumbent = ac.run()

            self.incumbent = self.incumbent.to_dict(orient='records')[0]

            print(f'\nBest Configuration found by {ac_tool} is:\n',
                  self.incumbent)

            return self.incumbent, None
        else:
            return None, None
