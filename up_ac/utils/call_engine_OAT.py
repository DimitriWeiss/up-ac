import dill
from unified_planning.io import PDDLReader
import sys
import os

# Read problem instance path
with open(sys.argv[1], 'r') as f:
    instance = f.read()

# Construct parameter dict
params = {}
for i, arg in enumerate(sys.argv):
    sys.argv[i] = arg.replace('{', '').replace('}', '')

for arg in sys.argv[2::2]:
    params[arg[1:]] = sys.argv[sys.argv.index(arg) + 1]
    
# Make sure modules are accesible
path = os.getcwd().rsplit('up_ac', 2)[0]
if path[-1] != "/":
    path += "/"
path += 'up_ac'
sys.path.append(r"{}".format(path))

# Load planner_feedback function
feedback = dill.load(open(f'{path}/OAT/feedback.pkl', 'rb'))
feedback = feedback(params, instance, PDDLReader())

# Print feedback for OAT to parse
print('\n\n', feedback)
