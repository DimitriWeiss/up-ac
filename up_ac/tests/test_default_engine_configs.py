"""Test up AC implementation."""
from unified_planning.io import PDDLReader
import unified_planning as up
import multiprocessing as mp
import sys
import os
import unittest


# make sure test can be run from anywhere
path = os.getcwd().rsplit('up_ac', 1)[0]
path += '/up_ac'
if not os.path.isfile(sys.path[0] + '/configurators.py') and \
        'up_ac' in sys.path[0]:
    sys.path.insert(0, sys.path[0].rsplit('up_ac', 1)[0] + '/up_ac')

    
from up_ac.Irace_interface import IraceInterface
from up_ac.Irace_configurator import IraceConfigurator

class TestDefaultConfigs(unittest.TestCase):

    def test_tamerConfig(self):
        engine = ['tamer']
        igaci = IraceInterface()
        igaci.read_engine_pcs(engine, f'{path}/engine_pcs')
        up.shortcuts.get_environment().credits_stream = None
        default_config = igaci.engine_param_spaces[engine[0]].get_default_configuration()
        self.assertEqual(dict(default_config), {'heuristic':'hadd','weight':0.5},f"Default configuration of {engine[0]} does not match specified default configuration")

    def test_enhsp(self):
        engine = ["enhsp"]
        igaci = IraceInterface()
        igaci.read_engine_pcs(engine, f'{path}/engine_pcs')
        up.shortcuts.get_environment().credits_stream = None
        default_config = igaci.engine_param_spaces[engine[0]].get_default_configuration()
        self.assertEqual(dict(default_config), {'heuristic':'hadd','search_algorithm':'gbfs'},f"Default configuration of {engine[0]} does not match specified default configuration")

    def test_fast_downward(self):
        engine = ["fast-downward"]
        igaci = IraceInterface()
        igaci.read_engine_pcs(engine, f'{path}/engine_pcs')
        up.shortcuts.get_environment().credits_stream = None
        default_config = igaci.engine_param_spaces[engine[0]].get_default_configuration()
        print(default_config)
        self.assertEqual(dict(default_config), {'cost_type': 'normal', 'fast_downward_search_config': 'astar', 'evaluator': 'blind', 'pruning': 'null'},f"Default configuration of {engine[0]} does not match specified default configuration")

    def test_lpg(self):
        engine = ["lpg"]
        igaci = IraceInterface()
        igaci.read_engine_pcs(engine, f'{path}/engine_pcs')
        up.shortcuts.get_environment().credits_stream = None
        default_config = igaci.engine_param_spaces[engine[0]].get_default_configuration()
        self.assertEqual(dict(default_config), {'avoid_best_action_cycles': '0', 'bestfirst': '1', 'choose_min_numA_fact': '1'},f"Default configuration of {engine[0]} does not match specified default configuration")

    def test_pyperplan(self):
        engine = ["pyperplan"]
        igaci = IraceInterface()
        igaci.read_engine_pcs(engine, f'{path}/engine_pcs')
        up.shortcuts.get_environment().credits_stream = None
        default_config = igaci.engine_param_spaces[engine[0]].get_default_configuration()
        self.assertEqual(dict(default_config), {'search':'astar'},f"Default configuration of {engine[0]} does not match specified default configuration")

if __name__ == '__main__':
    unittest.main()

