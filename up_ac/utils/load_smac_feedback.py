import dill
import os
import sys
from unified_planning.io import PDDLReader

reader = PDDLReader()


def get_feedback(config, instance, seed=0):

    try:
        import up_ac
        path = '/' + os.path.abspath(up_ac.__file__).strip('/__init__.py')
        path += '/utils'
    except ImportError:
        path = os.getcwd().rsplit('up_ac', 1)[0]
        if path[-1] != "/":
            path += "/"
        path += 'up_ac/utils'
    sys.path.append(r"{}".format(path))
    reader._env.error_used_name = False

    fb = \
        dill.load(open(f'{path}/feedback.pkl', 'rb'))

    feedback = fb(config, instance, seed, reader)

    return feedback
