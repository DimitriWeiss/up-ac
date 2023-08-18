"""Test up AC implementation."""
import unified_planning as up
import multiprocessing as mp
import sys
import os

# make sure test can be run from anywhere
path = os.getcwd().rsplit('up-ac', 1)[0]
path += 'up-ac'
if not os.path.isfile(sys.path[0] + '/configurators.py') and \
        'up-ac' in sys.path[0]:
    sys.path.insert(0, sys.path[0].rsplit('up-ac', 1)[0] + 'up-ac')

from Smac_configurator import SmacConfigurator
from Smac_interface import SmacInterface

# pddl instance to test with
instances = [f'{path}/test_problems/depot/problem.pddl',
             f'{path}/test_problems/citycar/problem.pddl',
             f'{path}/test_problems/sailing/problem.pddl',
             f'{path}/test_problems/visit_precedence/problem.pddl']

# test setting
engine = ['pyperplan']

metrics = ['quality', 'runtime']

# initialize generic Algorithm Configuration interface
sgaci = SmacInterface()
sgaci.read_engine_pcs(engine, f'{path}/engine_pcs')

# compute pddl instance features
instance_features = {}
for instance in instances:
    instance_features[instance] \
        = sgaci.compute_instance_features(
            instance.rsplit('/', 1)[0] + '/domain.pddl',
            instance)

up.shortcuts.get_environment().credits_stream = None

if __name__ == '__main__':
    mp.freeze_support()

    # Try optimizing for quality and runtime separately
    for metric in metrics:

        # initialize algorithm configurator
        SAC = SmacConfigurator()
        SAC.get_instance_features(instance_features)
        SAC.set_training_instance_set(instances)
        SAC.set_test_instance_set(instances)
        SAC_fb_func = SAC.get_feedback_function(sgaci, engine[0],
                                                metric, 'OneshotPlanner')
        
        # In case optimization of metric not possible with this engine
        if SAC_fb_func is None:
            print('There is no feedback function!')
            continue
        SAC.set_scenario('SMAC', engine[0],
                         sgaci.engine_param_spaces[engine[0]],
                         sgaci, configuration_time=60, n_trials=30,
                         min_budget=1, max_budget=3, crash_cost=0,
                         planner_timelimit=15, n_workers=1,
                         instance_features=SAC.instance_features)

        # Test feedback function
        default_config = \
            sgaci.engine_param_spaces[engine[0]].get_default_configuration()
        SAC_fb_func(default_config, instances[0])

        # run algorithm configuration
        incumbent, _ = SAC.optimize('SMAC', feedback_function=SAC_fb_func)

        # check configurations performance
        perf = SAC.evaluate('SMAC', metric, engine[0], 'OneshotPlanner',
                            SAC.incumbent, sgaci, planner_timelimit=15,
                            instances=[instances[0]])
        # save best configuration found
        SAC.save_config('.', SAC.incumbent, sgaci, engine[0])
