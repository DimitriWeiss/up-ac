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
from up_ac.OAT_configurator import OATConfigurator
from up_ac.OAT_interface import OATInterface
from up_ac.Smac_configurator import SmacConfigurator
from up_ac.Smac_interface import SmacInterface
from up_ac.utils.download_OAT import get_OAT, delete_OAT

with open(f"{path}/utils/download_OAT.py") as f:
    exec(f.read())

class TestEngines(unittest.TestCase):
# test setting
    def test_tamerIrace(self):
        engine = ['tamer']
        igaci = IraceInterface()
        igaci.read_engine_pcs(engine, f'{path}/engine_pcs')
        up.shortcuts.get_environment().credits_stream = None
        default_config = igaci.engine_param_spaces[engine[0]].get_default_configuration()
        self.assertEqual(dict(default_config), {'heuristic':'hadd','weight':0.5},f"Default configuration of {engine[0]} does not match specified default configuration")

    def test_tamerOAT(self):
        engine = ['tamer']
        get_OAT()
        ogaci = OATInterface()
        ogaci.read_engine_pcs(engine, f'{path}/engine_pcs')
        up.shortcuts.get_environment().credits_stream = None
        default_config = ogaci.engine_param_spaces[engine[0]].get_default_configuration()
        self.assertEqual(dict(default_config), {'heuristic':'hadd','weight':0.5},f"Default configuration of {engine[0]} does not match specified default configuration")
        delete_OAT()
        
    def test_tamerSmac(self):
        engine = ['tamer']
        Sgaci = SmacInterface()
        Sgaci.read_engine_pcs(engine, f'{path}/engine_pcs')
        up.shortcuts.get_environment().credits_stream = None
        default_config = Sgaci.engine_param_spaces[engine[0]].get_default_configuration()
        self.assertEqual(dict(default_config), {'heuristic':'hadd','weight':0.5},f"Default configuration of {engine[0]} does not match specified default configuration")

if __name__ == '__main__':
    unittest.main()