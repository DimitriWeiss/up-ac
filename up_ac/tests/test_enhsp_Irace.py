"""Test IraceConfigurator class and functions"""

import unified_planning as up
import sys
import os
import unittest

# make sure test can be run from anywhere
path = os.getcwd().rsplit('up-ac', 1)[0]
path += 'up-ac/up_ac'
if not os.path.isfile(sys.path[0] + '/configurators.py') and \
        'up-ac' in sys.path[0]:
    sys.path.insert(0, sys.path[0].rsplit('up-ac', 1)[0] + '/up-ac')

from up_ac.Irace_configurator import IraceConfigurator
from up_ac.Irace_interface import IraceInterface


class TestIraceEnhspOnQuality(unittest.TestCase):
    # pddl instance to test with
    instances = [f'{path}/test_problems/depot/problem.pddl',
                 f'{path}/test_problems/counters/problem.pddl',
                 f'{path}/test_problems/citycar/problem.pddl',
                 f'{path}/test_problems/sailing/problem.pddl',
                 f'{path}/test_problems/safe_road/problem.pddl']

    engine = ["enhsp"]

    igaci = IraceInterface()
    igaci.read_engine_pcs(engine, f'{path}/engine_pcs')

    metric = "quality"

    IAC = IraceConfigurator()
    IAC.set_training_instance_set(instances)
    IAC.set_test_instance_set(instances)

    IAC.set_scenario('irace', engine[0],
                     igaci.engine_param_spaces[engine[0]], igaci,
                     configuration_time=300, n_trials=30,
                     crash_cost=0, min_budget=3,
                     planner_timelimit=5, n_workers=3,
                     instance_features=None)
    IAC_fb_func = IAC.get_feedback_function(igaci, engine[0],
                                            metric, 'OneshotPlanner')

    def test_fb_func(self, IAC_fb_func=IAC_fb_func):
        self.assertIsNotNone(IAC_fb_func, "Operational mode not supported")

    def test_enhsp(self, IAC=IAC,
                   IAC_fb_func=IAC_fb_func, igaci=igaci,
                   metric=metric, engine=engine, instances=instances):

        default_config = \
            igaci.engine_param_spaces[engine[0]].get_default_configuration()
        experiment = {'id.instance': 1, 'configuration': default_config}
        IAC_fb_func(experiment, IAC.scenario)

    def test_evaluate(self, IAC=IAC, instances=instances, igaci=igaci,
                      engine=engine, metric=metric):

        if (IAC.incumbent is None) or (len(instances) == 0):
            self.assertIsNone(IAC.evaluate('irace', metric, engine[0], 'OneshotPlanner',
                                           IAC.incumbent, igaci),
                              "No incumbent and/or no instances should return None")
        else:
            self.assertIsNotNone(IAC.evaluate('irace', metric, engine[0], 'OneshotPlanner',
                                              IAC.incumbent, igaci), "Should have evaluated")


class TestIraceEnhspOnRuntime(unittest.TestCase):
    # pddl instance to test with
    instances = [f'{path}/test_problems/depot/problem.pddl',
                 f'{path}/test_problems/counters/problem.pddl',
                 f'{path}/test_problems/citycar/problem.pddl',
                 f'{path}/test_problems/sailing/problem.pddl',
                 f'{path}/test_problems/safe_road/problem.pddl']

    engine = ["enhsp"]

    igaci = IraceInterface()
    igaci.read_engine_pcs(engine, f'{path}/engine_pcs')

    metric = "quality"

    IAC = IraceConfigurator()
    IAC.set_training_instance_set(instances)
    IAC.set_test_instance_set(instances)

    IAC.set_scenario('irace', engine[0],
                     igaci.engine_param_spaces[engine[0]], igaci,
                     configuration_time=300, n_trials=30,
                     crash_cost=0, min_budget=3,
                     planner_timelimit=5, n_workers=3,
                     instance_features=None)
    IAC_fb_func = IAC.get_feedback_function(igaci, engine[0],
                                            metric, 'OneshotPlanner')

    def test_fb_func(self, IAC_fb_func=IAC_fb_func):
        self.assertIsNotNone(IAC_fb_func, "Operational mode not supported")

    def test_enhsp(self, IAC=IAC,
                   IAC_fb_func=IAC_fb_func, igaci=igaci,
                   metric=metric, engine=engine, instances=instances):

        default_config = \
            igaci.engine_param_spaces[engine[0]].get_default_configuration()
        experiment = {'id.instance': 1, 'configuration': default_config}
        IAC_fb_func(experiment, IAC.scenario)

    def test_evaluate(self, IAC=IAC, instances=instances, igaci=igaci,
                      engine=engine, metric=metric):

        if (IAC.incumbent is None) or (len(instances) == 0):
            self.assertIsNone(IAC.evaluate('irace', metric, engine[0], 'OneshotPlanner',
                                           IAC.incumbent, igaci),
                              "No incumbent and/or no instances should return None")
        else:
            self.assertIsNotNone(IAC.evaluate('irace', metric, engine[0], 'OneshotPlanner',
                                              IAC.incumbent, igaci), "Should have evaluated")


up.shortcuts.get_environment().credits_stream = None

if __name__ == '__main__':
    unittest.main()