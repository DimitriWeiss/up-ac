"""Functionalities for managing and calling configurators."""
from irace import irace
from unified_planning.exceptions import UPProblemDefinitionError, UPException
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
        Generate a function to run the planning engine and obtain feedback.

        Parameters:
            gaci (object): AC interface object.
            engine (str): Name of the planning engine.
            metric (str): Metric type, either 'runtime' or 'quality'.
            mode (str): Type of planning mode.
            gray_box (bool, optional): True if using a gray box approach.

        Returns:
            function: A function to provide feedback based on the specified parameters.

        Raises:
            ValueError: If the provided engine, metric, or mode is not supported.

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
                        UPProblemDefinitionError, UPException,
                        UnicodeDecodeError) as err:
                    print('\n** Error in planning engine!', err)
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

    def set_scenario(self, engine, param_space, gaci,
                     configuration_time=120,
                     n_trials=400, min_budget=1, max_budget=3, crash_cost=0,
                     planner_timelimit=30, n_workers=1, instances=[],
                     instance_features=None, metric='runtime'):
        """
        Set up the algorithm configuration scenario.

        Parameters:
            engine (str): Name of the engine.
            param_space (ConfigSpace.ConfigurationSpace): The ConfigSpace object defining the parameter space.
            gaci (object): AC interface object.
            configuration_time (int, optional): Overall configuration time budget.
            n_trials (int, optional): Maximum number of engine evaluations.
            min_budget (int, optional): Minimum number of instances to use.
            max_budget (int, optional): Maximum number of instances to use.
            crash_cost (int, optional): The cost to use if the engine fails.
            planner_timelimit (int, optional): Maximum runtime per evaluation.
            n_workers (int, optional): Number of cores to utilize.
            instances (list, optional): List of problem instance paths.
            instance_features (dict, optional): Dictionary containing instance names and lists of features.
            metric (str, optional): Optimization metric, either 'runtime' or 'quality'.

        Raises:
            ValueError: If the provided metric is not supported.

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

    def optimize(self, feedback_function=None, gray_box=False):
        """
        Run the algorithm configuration process.

        Parameters:
            feedback_function (function, optional): A function to run the engine and obtain feedback.
            gray_box (bool, optional): True if using a gray box approach.

        Returns:
            tuple or None: A tuple containing:
                - dict: The best configuration found.
                - None: If there is no feedback function.

        """

        if feedback_function is not None:

            print('\nStarting Parameter optimization\n')
            ac = irace(self.scenario,
                       self.irace_param_space,
                       feedback_function)
            self.incumbent = ac.run()

            self.incumbent = self.incumbent.to_dict(orient='records')[0]

            print('\nBest Configuration found is:\n',
                  self.incumbent)

            return self.incumbent, None
        else:
            return None, None
