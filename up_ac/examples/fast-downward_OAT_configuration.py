"""Test up AC implementation."""
import sys
import os

# make sure test can be run from anywhere
path = os.getcwd().rsplit('up_ac', 1)[0]
path += 'up_ac'
if not os.path.isfile(sys.path[0] + '/configurators.py') and \
        'up_ac' in sys.path[0]:
    sys.path.insert(0, sys.path[0].rsplit('up_ac', 1)[0] + 'up_ac')

from OAT_configurator import OATConfigurator
from OAT_interface import OATInterface

# pddl instance to test with
instances = [f'{path}/test_problems/miconic/problem.pddl',
             f'{path}/test_problems/depot/problem.pddl',
             f'{path}/test_problems/safe_road/problem.pddl']

print(instances)

# test setting
engine = ['fast-downward']

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
        OAC_fb_func = OAC.get_feedback_function('OAT', ogaci, engine[0],
                                                metric, 'OneshotPlanner')
        # run algorithm configuration
        incumbent, _ = OAC.optimize('OAT', feedback_function=OAC_fb_func)
        # check configurations performance
        perf = OAC.evaluate('OAT', metric, engine[0], 'OneshotPlanner',
                            OAC.incumbent, ogaci)
        # save best configuration found
        OAC.save_config('.', OAC.incumbent, ogaci, engine[0])
