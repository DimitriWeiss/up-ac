import unittest
import unified_planning as up
import multiprocessing as mp
import sys
import os

# make sure test can be run from anywhere
path = os.getcwd().rsplit('up-ac', 1)[0]
path += 'up-ac/up_ac'
if not os.path.isfile(sys.path[0] + '/configurators.py') and \
        'up-ac' in sys.path[0]:
    sys.path.insert(0, sys.path[0].rsplit('up-ac', 1)[0] + '/up-ac')

from up_ac.Smac_configurator import SmacConfigurator
from up_ac.Smac_interface import SmacInterface


class TestSmacFastDownwardOnQuality(unittest.TestCase):
    # pddl instance to test with
    instances = [f'{path}/test_problems/miconic/problem.pddl',
                 f'{path}/test_problems/depot/problem.pddl',
                 f'{path}/test_problems/safe_road/problem.pddl']

    # test setting
    engine = ['fast-downward']

    # initialize generic Algorithm Configuration interface
    sgaci = SmacInterface()
    sgaci.read_engine_pcs(engine, f'{path}/engine_pcs')

    metric = "quality"

    # compute pddl instance features
    instance_features = {}
    for instance in instances:
        instance_features[instance] \
            = sgaci.compute_instance_features(
            instance.rsplit('/', 1)[0] + '/domain.pddl',
            instance)

    mp.freeze_support()

    # initialize algorithm configurator
    SAC = SmacConfigurator()
    SAC.get_instance_features(instance_features)
    SAC.set_training_instance_set(instances)
    SAC.set_test_instance_set(instances)
    SAC_fb_func = SAC.get_feedback_function(sgaci, engine[0],
                                            metric, 'OneshotPlanner')

    SAC.set_scenario('SMAC', engine[0],
                     sgaci.engine_param_spaces[engine[0]],
                     sgaci, configuration_time=30, n_trials=30,
                     min_budget=1, max_budget=3, crash_cost=0,
                     planner_timelimit=5, n_workers=3,
                     instance_features=SAC.instance_features)

    default_config = \
        sgaci.engine_param_spaces[engine[0]].get_default_configuration()

    def test_fb_func(self, SAC_fb_func=SAC_fb_func, sgaci=sgaci,
                     engine=engine, instances=instances,
                     default_config=default_config):

        self.assertIsNotNone(SAC_fb_func, "Operational mode not supported")
        self.assertIsNotNone(SAC_fb_func(default_config, instances[0]))

    def test_optimize(self, SAC=SAC, SAC_fb_func=SAC_fb_func,
                      default_config=default_config):
        incumbent, _ = SAC.optimize('SMAC', feedback_function=SAC_fb_func)
        self.assertIsInstance(incumbent, dict)
        self.assertNotEqual(incumbent, default_config)

    def test_evaluate(self, metric=metric, engine=engine,
                      SAC=SAC, sgaci=sgaci):
        perf = SAC.evaluate('SMAC', metric, engine[0], 'OneshotPlanner',
                            SAC.incumbent, sgaci, planner_timelimit=5)
        self.assertIsNone(perf)


class TestSmacFastDownwardOnRuntime(unittest.TestCase):
    # pddl instance to test with
    instances = [f'{path}/test_problems/miconic/problem.pddl',
                 f'{path}/test_problems/depot/problem.pddl',
                 f'{path}/test_problems/safe_road/problem.pddl']

    # test setting
    engine = ['fast-downward']

    # initialize generic Algorithm Configuration interface
    sgaci = SmacInterface()
    sgaci.read_engine_pcs(engine, f'{path}/engine_pcs')

    metric = "runtime"

    # compute pddl instance features
    instance_features = {}
    for instance in instances:
        instance_features[instance] \
            = sgaci.compute_instance_features(
            instance.rsplit('/', 1)[0] + '/domain.pddl',
            instance)

    mp.freeze_support()

    # initialize algorithm configurator
    SAC = SmacConfigurator()
    SAC.get_instance_features(instance_features)
    SAC.set_training_instance_set(instances)
    SAC.set_test_instance_set(instances)
    SAC_fb_func = SAC.get_feedback_function(sgaci, engine[0],
                                            metric, 'OneshotPlanner')

    SAC.set_scenario('SMAC', engine[0],
                     sgaci.engine_param_spaces[engine[0]],
                     sgaci, configuration_time=30, n_trials=30,
                     min_budget=1, max_budget=3, crash_cost=0,
                     planner_timelimit=5, n_workers=3,
                     instance_features=SAC.instance_features)

    default_config = \
        sgaci.engine_param_spaces[engine[0]].get_default_configuration()

    def test_fb_func(self, SAC_fb_func=SAC_fb_func, sgaci=sgaci,
                     engine=engine, instances=instances,
                     default_config=default_config):

        self.assertIsNotNone(SAC_fb_func, "Operational mode not supported")
        self.assertIsNotNone(SAC_fb_func(default_config, instances[0]))

    def test_optimize(self, SAC=SAC, SAC_fb_func=SAC_fb_func,
                      default_config=default_config):
        incumbent, _ = SAC.optimize('SMAC', feedback_function=SAC_fb_func)
        self.assertIsInstance(incumbent, dict)
        self.assertNotEqual(incumbent, default_config)

    def test_evaluate(self, metric=metric, engine=engine,
                      SAC=SAC, sgaci=sgaci):
        perf = SAC.evaluate('SMAC', metric, engine[0], 'OneshotPlanner',
                            SAC.incumbent, sgaci, planner_timelimit=5)
        self.assertIsNone(perf)


up.shortcuts.get_environment().credits_stream = None

if __name__ == '__main__':
    unittest.main()
