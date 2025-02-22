"""Test up AC implementation."""
import unified_planning as up
import sys
import os
import unittest
from contextlib import redirect_stdout
import io

# make sure test can be run from anywhere
path = os.getcwd().rsplit('up_ac', 1)[0]
path += '/up_ac'
if not os.path.isfile(sys.path[0] + '/configurators.py') and \
        'up_ac' in sys.path[0]:
    sys.path.insert(0, sys.path[0].rsplit('up_ac', 1)[0] + '/up_ac')

from up_ac.Smac_configurator import SmacConfigurator
from up_ac.Smac_interface import SmacInterface

allowed_combinations = {'quality': {'OneshotPlanner': [['lpg'], ['fast-downward'], ['enhsp'], ['symk'], ['tamer'], ['pyperplan']], 
                                    'AnytimePlanner': [['fast-downward'], ['symk'], ['lpg-anytime'], ['enhsp-any']]},
                        'runtime': {'OneshotPlanner': [['lpg'], ['fast-downward'], ['enhsp'], ['symk'], ['tamer'], ['pyperplan']]}}
allowed_keys = allowed_combinations.keys()

disallowed_combinations = {'quality': {'OneshotPlanner': [['fmap']], 
                                    'AnytimePlanner': [['tamer'], ['pyperplan'], ['fmap']]}}
disallowed_keys = disallowed_combinations.keys()


up.shortcuts.get_environment().credits_stream = None


class TestDefaultConfigs(unittest.TestCase):

    def test_allowed_combinations(self):
        for key in allowed_keys:
            metric = key
            planners = allowed_combinations[key].keys()
            for planner in planners:
                engines = allowed_combinations[key][planner]
                for engine in engines:
                    sgaci = SmacInterface()
                    sgaci.read_engine_pcs(engine, f'{path}/engine_pcs')
                    SAC = SmacConfigurator()
                    trap = io.StringIO()
                    with redirect_stdout(trap): 
                        SAC_fb_func = SAC.get_feedback_function(sgaci, engine[0],
                                                                metric, planner)
                    self.assertFalse(SAC_fb_func == None, msg = f"There is no feedback function for the combination of metric:'{metric}', planner:'{planner}' and engine:'{engine[0]}' when there should be")


    def test_disallowed_combinations(self):
        for key in disallowed_keys:
            metric = key
            planners = disallowed_combinations[key].keys()
            for planner in planners:
                engines = disallowed_combinations[key][planner]
                for engine in engines:
                    sgaci = SmacInterface()
                    sgaci.read_engine_pcs(engine, f'{path}/engine_pcs')
                    SAC = SmacConfigurator()
                    trap = io.StringIO()
                    with redirect_stdout(trap): 
                        SAC_fb_func = SAC.get_feedback_function(sgaci, engine[0],
                                                                metric, planner)
                    self.assertTrue(SAC_fb_func == None, msg = f"There is a feedback function for the combination of metric:'{metric}', planner:'{planner}' and engine:'{engine[0]}' when there should not be")
                       
if __name__ == '__main__':
    unittest.main()