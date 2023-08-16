"""Test up AC implementation."""
import unified_planning as up
import sys
import os

# make sure test can be run from anywhere
path = os.getcwd().rsplit('up-ac', 2)[0]
path += 'up-ac'
if not os.path.isfile(sys.path[0] + '/configurators.py') and \
        'up-ac' in sys.path[0]:
    sys.path.insert(0, sys.path[0].rsplit('up-ac', 2)[0] + 'up-ac')

from Irace_configurator import IraceConfigurator
from Irace_interface import IraceInterface

# pddl instance to test with
instances = [f'{path}/test_problems/depot/problem.pddl',
             f'{path}/test_problems/counters/problem.pddl',
             f'{path}/test_problems/matchcellar/problem.pddl',
             f'{path}/test_problems/sailing/problem.pddl',
             f'{path}/test_problems/visit_precedence/problem.pddl']

# test setting
engine = ['tamer']

metrics = ['quality', 'runtime']

# initialize generic Algorithm Configuration interface
igaci = IraceInterface()
igaci.read_engine_pcs(engine, f'{path}/engine_pcs')

up.shortcuts.get_environment().credits_stream = None

if __name__ == '__main__':

    # Try optimizing for quality and runtime separately
    for metric in metrics:

        IAC = IraceConfigurator()
        IAC.set_training_instance_set(instances)
        IAC.set_test_instance_set(instances)
        IAC_fb_func = IAC.get_feedback_function(igaci, engine[0],
                                                metric, 'OneshotPlanner')

        # In case optimization of metric not possible with this engine
        if IAC_fb_func is None:
            print('There is no feedback function!')
            continue
        IAC.set_scenario('irace', engine[0],
                         igaci.engine_param_spaces[engine[0]], igaci,
                         configuration_time=300, n_trials=30,
                         crash_cost=0, min_budget=2,
                         planner_timelimit=5, n_workers=3,
                         instance_features=None)

        # Test feedback function
        default_config = \
            igaci.engine_param_spaces[engine[0]].get_default_configuration()
        experiment = {'id.instance': 1, 'configuration': default_config}
        IAC_fb_func(experiment, IAC.scenario)

        # run algorithm configuration
        incumbent, _ = IAC.optimize('irace', feedback_function=IAC_fb_func)
        # check configurations performance
        perf = IAC.evaluate('irace', metric, engine[0], 'OneshotPlanner',
                            IAC.incumbent, igaci)
        # save best configuration found
        IAC.save_config('.', IAC.incumbent, igaci, engine[0])
