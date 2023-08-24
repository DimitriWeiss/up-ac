"""Test up AC implementation."""
from unified_planning.io import PDDLReader
import unified_planning as up
import multiprocessing as mp
import time
import sys
import os
import unittest


# make sure test can be run from anywhere
path = os.getcwd().rsplit('up-ac', 1)[0]
path += 'up-ac'
if not os.path.isfile(sys.path[0] + '/configurators.py') and \
        'up-ac' in sys.path[0]:
    sys.path.insert(0, sys.path[0].rsplit('up-ac', 1)[0] + 'up-ac')
    
from Irace_interface import IraceInterface
from Irace_configurator import IraceConfigurator
from OAT_configurator import OATConfigurator
from OAT_interface import OATInterface


class TestDefaultConfigs(unittest.TestCase):
# test setting

    def test_tamerConfig(self):
        engine = ['tamer']

        # initialize generic Algorithm Configuration interface
        igaci = IraceInterface()
        igaci.read_engine_pcs(engine, f'{path}/engine_pcs')

        up.shortcuts.get_environment().credits_stream = None
        default_config = igaci.engine_param_spaces[engine[0]].get_default_configuration()
        self.assertEqual(dict(default_config), {'heuristic':'hadd','weight':0.5})

    def test_enhsp(self):
        engine = ["enhsp"]
        igaci = IraceInterface()
        igaci.read_engine_pcs(engine, f'{path}/engine_pcs')
        up.shortcuts.get_environment().credits_stream = None
        default_config = igaci.engine_param_spaces[engine[0]].get_default_configuration()
        self.assertEqual(dict(default_config), {'heuristic':'hadd','search_algorithm':'gbfs'})

    def test_fast_downward(self):
        engine = ["fast-downward"]
        igaci = IraceInterface()
        igaci.read_engine_pcs(engine, f'{path}/engine_pcs')
        up.shortcuts.get_environment().credits_stream = None
        default_config = igaci.engine_param_spaces[engine[0]].get_default_configuration()
        self.assertEqual(dict(default_config), {'cost_type': 'normal', 'fast_downward_search_config': 'astar', 'evaluator': 'blind', 'pruning': 'null'})

    def test_lpg(self):
        engine = ["lpg"]
        ogaci = OATInterface()
        ogaci.read_engine_pcs(engine, f'{path}/engine_pcs')
        up.shortcuts.get_environment().credits_stream = None
        default_config = ogaci.engine_param_spaces[engine[0]].get_default_configuration()
        self.assertEqual(dict(default_config), {'avoid_best_action_cycles': '0', 'bestfirst': '1', 'choose_min_numA_fact': '1'})

    def test_pyperplan(self):
        engine = ["pyperplan"]
        ogaci = OATInterface()
        ogaci.read_engine_pcs(engine, f'{path}/engine_pcs')
        up.shortcuts.get_environment().credits_stream = None
        default_config = ogaci.engine_param_spaces[engine[0]].get_default_configuration()
        self.assertEqual(dict(default_config), {'search':'astar'})

if __name__ == '__main__':
    unittest.main()

