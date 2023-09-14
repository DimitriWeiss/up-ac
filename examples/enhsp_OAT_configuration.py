"""Test up AC implementation."""
import sys
import os

# make sure test can be run from anywhere
path = os.getcwd().rsplit('up-ac', 1)[0]
path += 'up-ac'
if not os.path.isfile(sys.path[0] + '/configurators.py') and \
        'up-ac' in sys.path[0]:
    sys.path.insert(0, sys.path[0].rsplit('up-ac', 1)[0] + 'up-ac')

from OAT_configurator import OATConfigurator
from OAT_interface import OATInterface

# pddl instance to test with
instances = [f'{path}/test_problems/depot/problem.pddl',
             f'{path}/test_problems/counters/problem.pddl',
             f'{path}/test_problems/citycar/problem.pddl',
             f'{path}/test_problems/sailing/problem.pddl',
             f'{path}/test_problems/safe_road/problem.pddl']

print(instances)

# test setting
engine = ['enhsp']

metrics = ['quality', 'runtime']

# initialize generic Algorithm Configuration interface
ogaci = OATInterface()
ogaci.read_engine_pcs(engine, f'{path}/engine_pcs')

if __name__ == '__main__':

    for metric in metrics:

        OAC = OATConfigurator()
        OAC.set_training_instance_set(instances)
        OAC.set_test_instance_set(instances)
        
        OAC.set_scenario(engine[0],
                         ogaci.engine_param_spaces[engine[0]], ogaci,
                         configuration_time=30, n_trials=30,
                         crash_cost=0, planner_timelimit=15, n_workers=3,
                         instance_features=None, popSize=5, metric=metric,
                         evlaLimit=1)
        OAC_fb_func = OAC.get_feedback_function(ogaci, engine[0],
                                                metric, 'OneshotPlanner')
        # run algorithm configuration
        # incumbent, _ = OAC.optimize('OAT', feedback_function=OAC_fb_func)
        # check configurations performance
        # print(OAC.incumbent)
        params = {'-heuristic': 'hadd', '-search_algorithm': 'gbfs'}
        #perf = OAC.evaluate('OAT', metric, engine[0], 'OneshotPlanner',
        #                    OAC.incumbent, ogaci)
        perf = OAC.evaluate('OAT', metric, engine[0], 'OneshotPlanner',
                            params, ogaci)
        # save best configuration found
        OAC.save_config('.', OAC.incumbent, ogaci, engine[0])
