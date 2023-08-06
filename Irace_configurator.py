"""Functionalities for managing and calling configurators."""
from irace import irace

from AC_interface import *
from configurators import Configurator

class Configurator(Configurator):
    """Configurator functions."""

    def __init__(self):
        """Initialize Irace configurator."""
        Configurator.__init__(self)

    def get_feedback_function(self, ac_tool, gaci, engine, metric, mode, gray_box=False):
        if engine in self.capabilities[metric][mode]:
            self.metric = metric
            def planner_feedback(experiment, scenario):
                start = timeit.default_timer()
                instance_p = scenario['instances'][experiment['id.instance'] - 1]
                domain_path = instance_p.rsplit('/', 1)[0]
                out_file = instance_p.rsplit('/', 1)[1]
                domain = f'{domain_path}/domain.pddl'
                pddl_problem = self.reader.parse_problem(f'{domain}',
                                                    f'{instance_p}')
                config = dict(experiment['configuration'])

                feedback = \
                    gaci.run_engine_config(ac_tool,
                                           config,
                                           metric,
                                           engine,
                                           mode,
                                           pddl_problem)

                runtime = timeit.default_timer() - start
                if feedback is not None:
                    # SMAC always minimizes
                    if metric == 'quality':
                        feedback = {'cost': -feedback, 'time': runtime}
                        return feedback
                    elif metric == 'runtime':
                        feedback = {'cost': runtime, 'time': runtime}
                        return feedback
                else:
                    if metric == 'runtime':
                        feedback = {'cost': runtime, 'time': runtime}
                    else:
                        feedback = feedback = {'cost': self.crash_cost, 'time': self.crash_cost}
                    return feedback

            return planner_feedback
        else:
            print(f'Algorithm Configuration for {metric} of {ac_tool} in {mode} is not supported.')
            return None

    def set_scenario(self, ac_tool, engine, param_space, gaci, configuration_time=120,
                     n_trials=400, min_budget=1, max_budget=3, crash_cost=0,
                     planner_timelimit=30, n_workers=1, instances=[],
                     instance_features=None, metric='runtime', popSize=128, evlaLimit=2147483647):
        if not instances:
            instances = self.train_set
        self.crash_cost = crash_cost
        default_conf = gaci.get_ps_irace(param_space)

        if metric == 'quality':
            test_type = 'friedman'
            capping = False
        elif metric == 'runtime':
            test_type = 't-test'
            capping = True
        # See https://mlopez-ibanez.github.io/irace/reference/defaultScenario.html
        scenario = dict(
            maxTime = configuration_time, # We want to optimize for <configuration_time> seconds
            instances = instances, # List of training instances
            debugLevel = 3, 
            digits = 10, # number of decimal places to be considered for the real parameters
            parallel=n_workers, # Number of parallel runs
            forbiddenFile = "forbidden.txt",
            logFile = "",
            initConfigurations=default_conf,
            nbConfigurations=8,
            deterministic = True,
            testType=test_type,
            capping=capping,
            boundMax=planner_timelimit
        )

        self.irace_param_space = gaci.irace_param_space

        print('\nIrace scenario is set.\n')

        self.scenario = scenario

    def optimize(self, ac_tool, feedback_function=None, gray_box=False):
        if feedback_function is not None:

            print('\nStarting Parameter optimization\n')
            ac = irace(self.scenario,
                self.irace_param_space,
                feedback_function
                )
            self.incumbent = ac.run()

            self.incumbent = self.incumbent.to_dict(orient='records')[0]

            print(f'\nBest Configuration found by {ac_tool} is:\n', self.incumbent)

            return self.incumbent, None
        else:
            return None, None
