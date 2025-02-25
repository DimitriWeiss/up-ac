"""Functionalities for managing and calling configurators."""
from irace import irace
from unified_planning.exceptions import UPProblemDefinitionError, UPException

import timeit
import signal
import sys
import os
from contextlib import contextmanager
from pebble import concurrent
from pebble.common import ProcessExpired
from concurrent.futures import TimeoutError

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from AC_interface import *
from configurators import Configurator


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


class IraceConfigurator(Configurator):
    """Configurator functions."""

    def __init__(self):
        """Initialize Irace configurator."""
        Configurator.__init__(self)

    def get_feedback_function(self, gaci, engine, metric, mode,
                              gray_box=False):
        """
        Generate a function to run the planning engine and obtain feedback.

        :param gaci: AC interface object.
        :type gaci: object
        :param engine: Name of the planning engine.
        :type engine: str
        :param metric: Metric type, either 'runtime' or 'quality'.
        :type metric: str
        :param mode: Type of planning mode.
        :type mode: str
        :param gray_box: True if using a gray box approach (optional).
        :type gray_box: bool, optional

        :return: A function to provide feedback based on the specified parameters.
        :rtype: function

        :raises ValueError: If the provided engine, metric, or mode is not supported.
        """
        if engine in self.capabilities[metric][mode]:
            self.metric = metric

            def planner_feedback(experiment, scenario):

                if metric == 'quality':
                    timelimit = self.planner_timelimit - self.patience
                else:
                    timelimit = self.planner_timelimit

                start = timeit.default_timer()
                instance_p = \
                    self.scenario['instances'][experiment['id.instance'] - 1]
                try:
                    if ',' in instance_p:
                        instance_p = eval(instance_p)
                        pddl_problem = \
                            self.reader.parse_problem(f'{instance_p[1]}',
                                                      f'{instance_p[0]}')
                    else:
                        domain_path = instance_p.rsplit('/', 1)[0]
                        domain = f'{domain_path}/domain.pddl'
                        pddl_problem = self.reader.parse_problem(f'{domain}',
                                                                 f'{instance_p}')
                    config = dict(experiment['configuration'])

                except UPProblemDefinitionError as e:
                    print(e)
                    if metric == 'runtime':
                        feedback = timelimit * 2
                        feedback = {'cost': feedback, 'time': feedback}
                    else:
                        feedback = self.crash_cost
                        feedback = {'cost': feedback, 'time': timelimit * 2}
                    return feedback

                '''
                feedback = \
                    gaci.run_engine_config(config,
                                           metric,
                                           engine,
                                           mode,
                                           pddl_problem)
                '''
                
                try:
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
                        if engine == 'tamer' or engine == 'pyperplan':
                            future = solve(config, metric, engine,
                                           mode, pddl_problem)
                            if metric == 'runtime':
                                penalty = timelimit
                            elif metric == 'quality':
                                penalty = self.crash_cost
                            try:
                                feedback = future.result()
                            except (TimeoutError, ProcessExpired) as err:
                                print(err)
                                feedback = penalty
                            except Exception as e:
                                print(e)
                                feedback = penalty
                        else:
                            with time_limit(timelimit):
                                feedback = solve(config, metric, engine,
                                                 mode, pddl_problem)

                    except TimeoutException:  # TimeoutError:
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
                    print('\n Unsolvable feedback:', feedback, '\n')
                    return {'cost': (self.crash_cost), 'time': timelimit}

                elif feedback is not None:
                    if metric == 'quality':
                        self.print_feedback(engine, instance_p, feedback)
                        runtime = timeit.default_timer() - start
                        feedback = {'cost': feedback, 'time': runtime}
                        return feedback
                    elif metric == 'runtime':
                        if engine in ('tamer', 'pyperplan') and feedback == 'measure':
                            feedback = timeit.default_timer() - start
                        self.print_feedback(engine, instance_p, feedback)
                        feedback = {'cost': feedback, 'time': feedback}
                        return feedback
                else:
                    if metric == 'runtime':
                        # Penalty is max runtime in runtime scenario
                        feedback = timelimit * 2
                        self.print_feedback(engine, instance_p, feedback)
                        feedback = {'cost': feedback, 'time': feedback}
                    else:
                        # Penalty is defined by user in quality scenario
                        feedback = self.crash_cost
                        self.print_feedback(engine, instance_p, feedback)
                        feedback = {'cost': feedback, 'time': timelimit * 2}
                    return feedback

            return planner_feedback
        else:
            if self.verbose:
                print(f'Algorithm Configuration for {metric} of {engine} in' + 
                      f' {mode} is not supported.')
            return None

    def set_scenario(self, engine, param_space, gaci,
                     configuration_time=120,
                     n_trials=400, min_budget=1, max_budget=3, crash_cost=0,
                     planner_timelimit=30, n_workers=1, instances=[],
                     instance_features=None, metric='runtime', patience=10):
        """
        Set up the algorithm configuration scenario.

        :param engine: Name of the engine.
        :type engine: str
        :param param_space: The ConfigSpace object defining the parameter space.
        :type param_space: ConfigSpace.ConfigurationSpace
        :param gaci: AC interface object.
        :type gaci: object
        :param configuration_time: Overall configuration time budget (optional).
        :type configuration_time: int, optional
        :param n_trials: Maximum number of engine evaluations (optional).
        :type n_trials: int, optional
        :param min_budget: Minimum number of instances to use (optional).
        :type min_budget: int, optional
        :param max_budget: Maximum number of instances to use (optional).
        :type max_budget: int, optional
        :param crash_cost: The cost to use if the engine fails (optional).
        :type crash_cost: int, optional
        :param planner_timelimit: Maximum runtime per evaluation (optional).
        :type planner_timelimit: int, optional
        :param n_workers: Number of cores to utilize (optional).
        :type n_workers: int, optional
        :param instances: List of problem instance paths (optional).
        :type instances: list, optional
        :param instance_features: Dictionary containing instance names and lists of features (optional).
        :type instance_features: dict, optional
        :param metric: Optimization metric, either 'runtime' or 'quality' (optional).
        :type metric: str, optional

        :raises ValueError: If the provided metric is not supported.
        """
        if not instances:
            if isinstance(self.train_set[0], tuple):
                instances = []
                for ts in self.train_set:
                    instances.append(str(ts))
            else:
                instances = self.train_set
        self.crash_cost = crash_cost
        self.patience = patience
        
        default_conf, forbiddens = gaci.get_ps_irace(param_space)

        if metric == 'quality':
            test_type = 'friedman'
            capping = False
            self.planner_timelimit = planner_timelimit + patience
        elif metric == 'runtime':
            test_type = 't-test'
            capping = True
            self.planner_timelimit = planner_timelimit
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
                # nbConfigurations=8,  # we let Irace decide
                deterministic=True,
                testType=test_type,
                capping=capping,
                boundMax=self.planner_timelimit,
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
                # nbConfigurations=8,  # we let Irace decide
                deterministic=True,
                testType=test_type,
                capping=capping,
                boundMax=self.planner_timelimit,
                firstTest=min_budget
            )

        self.irace_param_space = gaci.irace_param_space

        if self.verbose:
            print('\nIrace scenario is set.\n')

        self.scenario = scenario

    def optimize(self, feedback_function=None, gray_box=False):
        """
        Run the algorithm configuration process.

        :param feedback_function: A function to run the engine and obtain feedback (optional).
        :type feedback_function: function, optional
        :param gray_box: True if using a gray box approach (optional).
        :type gray_box: bool, optional

        :returns: A tuple containing:
            - dict: The best configuration found.
            - None: If there is no feedback function.
        :rtype: tuple or None
        """

        if feedback_function is not None:

            if self.verbose:
                print('\nStarting Parameter optimization\n')
            ac = irace(self.scenario,
                       self.irace_param_space,
                       feedback_function)
            self.incumbent = ac.run()

            self.incumbent = self.incumbent.to_dict(orient='records')[0]

            if self.verbose:
                print('\nBest Configuration found is:\n',
                      self.incumbent)

            return self.incumbent, None
        else:
            return None, None
