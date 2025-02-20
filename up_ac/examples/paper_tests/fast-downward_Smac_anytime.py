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
    sys.path.insert(0, sys.path[0].rsplit('up-ac', 1)[0] + '/up-ac')

from up_ac.Smac_configurator import SmacConfigurator
from up_ac.Smac_interface import SmacInterface

# Training Instance Set
train_instances = []
with open('instance_lists/tamer_train.txt', 'r') as f:
    for line in f:
        train_instances.append(eval(line.strip()))

# Test Instance Set
test_instances = []
with open('instance_lists/tamer_test.txt', 'r') as f:
    for line in f:
        test_instances.append(eval(line.strip()))

# test setting
engine = ['fast-downward']

metrics = ['quality']

# initialize generic Algorithm Configuration interface
sgaci = SmacInterface()
sgaci.read_engine_pcs(engine, f'{path}/engine_pcs')

# compute pddl instance features
instance_features = {}
for instance in train_instances:
    if isinstance(instance, tuple):
        domain = instance[1]
        instance = instance[0]
        instance_features[instance] \
            = sgaci.compute_instance_features(
                domain,
                instance)
    else:
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
        SAC.set_training_instance_set(train_instances)
        SAC.set_test_instance_set(test_instances)

        SAC.set_scenario(engine[0],
                         sgaci.engine_param_spaces[engine[0]],
                         sgaci, configuration_time=40000, n_trials=300,
                         min_budget=3, max_budget=5, crash_cost=0,
                         planner_timelimit=300, n_workers=4,
                         instance_features=SAC.instance_features,
                         output_dir='smac_output_test', metric=metric)

        SAC_fb_func = SAC.get_feedback_function(sgaci, engine[0],
                                                metric, 'AnytimePlanner')
        
        # In case optimization of metric not possible with this engine
        if SAC_fb_func is None:
            print('There is no feedback function!')
            continue

        # Test feedback function
        default_config = \
            sgaci.engine_param_spaces[engine[0]].get_default_configuration()

        # run algorithm configuration
        incumbent, _ = SAC.optimize(feedback_function=SAC_fb_func)

        # check configurations performance
        #perf = SAC.evaluate(metric, engine[0], 'OneshotPlanner',
        #                    SAC.incumbent, sgaci, planner_timelimit=5)

        # save best configuration found
        #SAC.save_config('.', SAC.incumbent, sgaci, engine[0])
