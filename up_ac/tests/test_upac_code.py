"""Test up AC implementation."""
from unified_planning.io import PDDLReader
import sys
import os

path = os.getcwd().rsplit('up_ac', 2)[0]
path += '/up_ac'

if not os.path.isfile(sys.path[0] + '/configurators.py') and \
        'up_ac' in sys.path[0]:
    sys.path.insert(0, sys.path[0].rsplit('up_ac', 2)[0] + '/up_ac')

from up_ac.AC_interface import GenericACInterface

gaci = GenericACInterface()

print('Available engines:\n', gaci.available_engines, '\n')

engines = ['lpg', 'fast-downward', 'enhsp']

gaci.read_engine_pcs(engines, f'{path}/engine_pcs')

default_param = \
    gaci.engine_param_spaces[engines[0]].get_default_configuration()

reader = PDDLReader()
pddl_problem = reader.parse_problem(f'{path}/test_problems/depot/domain.pddl',
                                    f'{path}/test_problems/depot/problem.pddl')

metrics = ['quality', 'runtime']

for metric in metrics:

    feedback = \
        gaci.run_engine_config(default_param,
                               metric,
                               engines[0],
                               'OneshotPlanner',
                               pddl_problem)

    print('1. Feedback:\n\n', feedback)

    default_param = \
        gaci.engine_param_spaces[engines[1]].get_default_configuration()

    feedback = \
        gaci.run_engine_config(default_param,
                               metric,
                               engines[1],
                               'OneshotPlanner',
                               pddl_problem)

    print('2. Feedback:\n\n', feedback)

    default_param = \
        gaci.engine_param_spaces[engines[2]].get_default_configuration()

    feedback = \
        gaci.run_engine_config(default_param,
                               metric,
                               engines[2],
                               'OneshotPlanner',
                               pddl_problem)

    print('3. Feedback:\n\n', feedback)
