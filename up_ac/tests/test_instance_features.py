"""Test pddl instance feature computation implementation."""
import sys
import os

path = os.getcwd().rsplit('up_ac', 2)[0]
path += '/up_ac'

if not os.path.isfile(sys.path[0] + '/configurators.py') \
        and 'up_ac' in sys.path[0]:
    sys.path.insert(0, sys.path[0].rsplit('up_ac', 2)[0] + 'up_ac')

from up_ac.AC_interface import GenericACInterface

gaci = GenericACInterface()

test_problems = path + '/test_problems'
features = {}

for file in os.listdir(test_problems):
    if os.path.isfile(test_problems + '/' + file + '/problem.pddl'):
        features[test_problems + '/' + file + '/problem.pddl'] \
            = gaci.compute_instance_features(
                test_problems + '/' + file + '/domain.pddl',
                test_problems + '/' + file + '/problem.pddl')

print(features)
