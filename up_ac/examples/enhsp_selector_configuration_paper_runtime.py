"""Test up AC implementation."""
import unified_planning as up
import multiprocessing as mp
import sys
import os

# make sure test can be run from anywhere
path = os.getcwd().rsplit('up-ac', 1)[0]
if path[-1] != "/":
    path += "/"
path += 'up-ac/up_ac'
if not os.path.isfile(sys.path[0] + '/configurators.py') and \
        'up-ac' in sys.path[0]:
    sys.path.append(sys.path[0].rsplit('up-ac', 1)[0] + 'up-ac')
    sys.path.append(sys.path[0].rsplit('up-ac', 1)[0] + 'up-ac/up_ac')
    sys.path.append(sys.path[0].rsplit('up-ac', 1)[0] + 'up-ac/utils')

from up_ac.Selector_configurator import SelectorConfigurator
from up_ac.Selector_interface import SelectorInterface

# Training Instance Set
train_instances = []
with open('paper_instances_enhsp_test_short.txt', 'r') as f:
    for line in f:
        train_instances.append(eval(line.strip()))

# Test Instance Set
test_instances = []
with open('paper_instances_enhsp_test_short.txt', 'r') as f:
    for line in f:
        test_instances.append(eval(line.strip()))

train_instances = [f'{path}/test_problems/visit_precedence/problem.pddl',
             f'{path}/test_problems/counters/problem.pddl',
             f'{path}/test_problems/depot/problem.pddl',
             f'{path}/test_problems/miconic/problem.pddl',
             f'{path}/test_problems/matchcellar/problem.pddl']

'''
enhsp_ANY_INSTS = [('/media/dweiss/Transcend7/AIPlan4EU/Instances/IPC2014/TESTING/seq-sat/CaveDiving/testing05A_easy.pddl', '/media/dweiss/Transcend7/AIPlan4EU/Instances/IPC2014/TESTING/seq-sat/CaveDiving/domain.pddl'),
('/media/dweiss/Transcend7/AIPlan4EU/Instances/IPC2014/TESTING/seq-sat/CaveDiving/testing06A_easy.pddl', '/media/dweiss/Transcend7/AIPlan4EU/Instances/IPC2014/TESTING/seq-sat/CaveDiving/domain.pddl'),
('/media/dweiss/Transcend7/AIPlan4EU/Instances/IPC2014/TESTING/seq-sat/Transport/p19.pddl', '/media/dweiss/Transcend7/AIPlan4EU/Instances/IPC2014/TESTING/seq-sat/Transport/domain.pddl'),
('/media/dweiss/Transcend7/AIPlan4EU/Instances/ipc2018-classical-domains-89741bb4cdd0/sat/agricola/p07.pddl', '/media/dweiss/Transcend7/AIPlan4EU/Instances/ipc2018-classical-domains-89741bb4cdd0/sat/agricola/domain.pddl'),
('/media/dweiss/Transcend7/AIPlan4EU/Instances/ipc2023-dataset-main/sat/folding/p06.pddl', '/media/dweiss/Transcend7/AIPlan4EU/Instances/ipc2023-dataset-main/sat/folding/domain.pddl')]

train_instances = enhsp_ANY_INSTS
'''
test_instances = train_instances

#train_instances = [f'{path}/test_problems/visit_precedence/problem.pddl',
#                   f'{path}/test_problems/counters/problem.pddl',
#                   f'{path}/test_problems/depot/problem.pddl']

train_instances = [f'{path}/test_problems/miconic/problem.pddl',
                   f'{path}/test_problems/depot/problem.pddl',
                   f'{path}/test_problems/safe_road/problem.pddl']

test_instances = train_instances

# test setting
engine = ['fast-downward']

metrics = ['quality']  # ['quality', 'runtime']

# initialize generic Algorithm Configuration interface
selgaci = SelectorInterface()
selgaci.read_engine_pcs(engine, f'{path}/engine_pcs')

# compute pddl instance features
instance_features = {}
for instance in train_instances:
    if isinstance(instance, tuple):
        domain = instance[1]
        instance = instance[0]
        instance_features[instance] \
            = selgaci.compute_instance_features(
                domain,
                instance)
    else:
        instance_features[instance] \
            = selgaci.compute_instance_features(
                instance.rsplit('/', 1)[0] + '/domain.pddl',
                instance)


up.shortcuts.get_environment().credits_stream = None
crash_cost = 10000


if __name__ == '__main__':

    # Try optimizing for quality and runtime separately
    for metric in metrics:

        # initialize algorithm configurator
        SelAC = SelectorConfigurator()
        SelAC.get_instance_features(instance_features)
        SelAC.set_training_instance_set(train_instances)
        SelAC.set_test_instance_set(test_instances)

        SelAC.set_scenario(engine[0],
                           selgaci.engine_param_spaces[engine[0]],
                           selgaci, configuration_time=3600, tourn_size=2,
                           min_budget=2, max_budget=3, crash_cost=crash_cost,
                           planner_timelimit=300, n_workers=4, 
                           instance_features=SelAC.instance_features,
                           output_dir='enhsp_selector', metric=metric)
  
        SelAC_fb_func = SelAC.get_feedback_function(selgaci, engine[0],
                                                    metric, 'AnytimePlanner')

        # In case optimization of metric not possible with this engine
        if SelAC_fb_func is None:
            print('There is no feedback function!')
            continue

        # Test feedback function
        default_config = \
            selgaci.engine_param_spaces[engine[0]].get_default_configuration()

        # run algorithm configuration
        incumbent, _ = SelAC.optimize(feedback_function=SelAC_fb_func)

        # check configurations performance
        #perf = SAC.evaluate(metric, engine[0], 'OneshotPlanner',
        #                    SAC.incumbent, sgaci, planner_timelimit=5)

        # save best configuration found
        #SAC.save_config('.', SAC.incumbent, sgaci, engine[0])
