"""Test up AC implementation."""
import unified_planning as up
import sys
import os

# make sure test can be run from anywhere
path = os.getcwd().rsplit('up-ac', 1)[0]
if path[-1] != "/":
    path += "/"
path += 'up-ac/up_ac'
if not os.path.isfile(sys.path[0] + '/configurators.py') and \
        'up-ac' in sys.path[0]:
    sys.path.insert(0, sys.path[0].rsplit('up-ac', 1)[0] + '/up-ac')

from up_ac.Irace_configurator import IraceConfigurator
from up_ac.Irace_interface import IraceInterface

# pddl instance to test with
instances = [f'{path}/test_problems/miconic/problem.pddl',
             f'{path}/test_problems/depot/problem.pddl',
             f'{path}/test_problems/safe_road/problem.pddl']

# test setting
engine = ['fast-downward']

metrics = ['quality', 'runtime']

# initialize generic Algorithm Configuration interface
igaci = IraceInterface()
igaci.read_engine_pcs(engine, f'{path}/engine_pcs')

up.shortcuts.get_environment().credits_stream = None
crash_cost = sys.maxsize - 1

if __name__ == '__main__':

    # Try optimizing for quality and runtime separately
    for metric in metrics:

        IAC = IraceConfigurator()
        IAC.set_training_instance_set(instances)
        IAC.set_test_instance_set(instances)

        IAC.set_scenario(engine[0],
                         igaci.engine_param_spaces[engine[0]], igaci,
                         configuration_time=60, n_trials=30,
                         crash_cost=crash_cost, min_budget=2,
                         planner_timelimit=5, n_workers=3,
                         instance_features=None)

        IAC_fb_func = IAC.get_feedback_function(igaci, engine[0],
                                                metric, 'OneshotPlanner')

        # In case optimization of metric not possible with this engine
        if IAC_fb_func is None:
            print('There is no feedback function!')
            continue

        # Test feedback function
        default_config = \
            igaci.engine_param_spaces[engine[0]].get_default_configuration()
        experiment = {'id.instance': 1, 'configuration': default_config}
        IAC_fb_func(experiment, IAC.scenario)

        # run algorithm configuration
        incumbent, _ = IAC.optimize(feedback_function=IAC_fb_func)
        # check configurations performance
        perf = IAC.evaluate(metric, engine[0], 'OneshotPlanner',
                            IAC.incumbent, igaci)
        # save best configuration found
        IAC.save_config('.', IAC.incumbent, igaci, engine[0])
