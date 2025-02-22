import unittest
import unified_planning as up
import sys
import os

# make sure test can be run from anywhere
path = os.getcwd().rsplit('up-ac', 1)[0]
path += 'up-ac/up_ac'
if not os.path.isfile(sys.path[0] + '/configurators.py') and \
        'up-ac' in sys.path[0]:
    sys.path.insert(0, sys.path[0].rsplit('up-ac', 1)[0] + '/up-ac')

from up_ac.OAT_configurator import OATConfigurator
from up_ac.OAT_interface import OATInterface
from up_ac.utils.download_OAT import get_OAT, delete_OAT, copy_call_engine_OAT

get_OAT()
copy_call_engine_OAT()


class TestOatLpgOnQuality(unittest.TestCase):

    get_OAT()
    copy_call_engine_OAT()
    # pddl instance to test with
    instances = [f'{path}/test_problems/visit_precedence/problem.pddl',
                 f'{path}/test_problems/counters/problem.pddl',
                 f'{path}/test_problems/depot/problem.pddl',
                 f'{path}/test_problems/miconic/problem.pddl',
                 f'{path}/test_problems/matchcellar/problem.pddl']

    # test setting
    engine = ['lpg']
    metric = "quality"

    ogaci = OATInterface()
    ogaci.read_engine_pcs(engine, f'{path}/engine_pcs')

    OAC = OATConfigurator()
    OAC.set_training_instance_set(instances)
    OAC.set_test_instance_set(instances)

    OAC.set_scenario(engine[0], ogaci.engine_param_spaces[engine[0]],
                     ogaci, configuration_time=30, n_trials=30,
                     crash_cost=10000, planner_timelimit=15, n_workers=3,
                     instance_features=None, popSize=5, metric=metric,
                     evalLimit=1)

    OAC_fb_func = OAC.get_feedback_function(ogaci, engine[0],
                                        metric, 'OneshotPlanner')

    def test_A_fb_func(self, OAC_fb_func=OAC_fb_func):
        self.assertIsNotNone(OAC_fb_func, "Operational mode not supported")

    def test_B_optimize(self, OAC=OAC, OAC_fb_func=OAC_fb_func):
        incumbent, _ = OAC.optimize(feedback_function=OAC_fb_func)
        self.assertIsInstance(incumbent, dict)

    def test_C_evaluate(self, OAC=OAC, metric=metric,
                      engine=engine, ogaci=ogaci):

        perf = OAC.evaluate(metric, engine[0], 'OneshotPlanner',
                            OAC.incumbent, ogaci)
        if OAC.incumbent is None:
            self.assertIsNone(perf, "No incumbent and/or no instances should return None")
        else:
            self.assertIsNotNone(perf, "Should have evaluated")


class TestOatLpgOnRuntime(unittest.TestCase):
    # pddl instance to test with
    instances = [f'{path}/test_problems/visit_precedence/problem.pddl',
                 f'{path}/test_problems/counters/problem.pddl',
                 f'{path}/test_problems/depot/problem.pddl',
                 f'{path}/test_problems/miconic/problem.pddl',
                 f'{path}/test_problems/matchcellar/problem.pddl']

    # test setting
    engine = ['lpg']
    metric = "runtime"

    ogaci = OATInterface()
    ogaci.read_engine_pcs(engine, f'{path}/engine_pcs')

    OAC = OATConfigurator()
    OAC.set_training_instance_set(instances)
    OAC.set_test_instance_set(instances)

    OAC.set_scenario(engine[0],
                     ogaci.engine_param_spaces[engine[0]], ogaci,
                     configuration_time=30, n_trials=30,
                     crash_cost=0, planner_timelimit=15, n_workers=3,
                     instance_features=None, popSize=5, metric=metric,
                     evalLimit=1)

    OAC_fb_func = OAC.get_feedback_function(ogaci, engine[0],
                                        metric, 'OneshotPlanner')

    def test_A_fb_func(self, OAC_fb_func=OAC_fb_func):
        self.assertIsNotNone(OAC_fb_func, "Operational mode not supported")

    def test_B_optimize(self, OAC=OAC, OAC_fb_func=OAC_fb_func):
        incumbent, _ = OAC.optimize(feedback_function=OAC_fb_func)
        self.assertIsInstance(incumbent, dict)

    def test_C_evaluate(self, OAC=OAC, metric=metric,
                        engine=engine, ogaci=ogaci):

        perf = OAC.evaluate(metric, engine[0], 'OneshotPlanner',
                            OAC.incumbent, ogaci)
        if OAC.incumbent is None:
            self.assertIsNone(perf, "No incumbent and/or no instances should return None")
        else:
            self.assertIsNotNone(perf, "Should have evaluated")

    def test_X_delete_OAT(self):
        delete_OAT()
        path = os.getcwd().rsplit('up_ac', 1)[0]
        if path[-1] != "/":
            path += "/"
        path += 'up_ac'
        self.assertFalse(os.path.isdir(f'{path}/OAT/'))


up.shortcuts.get_environment().credits_stream = None
if __name__ == '__main__':
    unittest.main()