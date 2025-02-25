"""Wrapper to call UP planning engines from selector."""

import argparse
import ast
import os


class UP_AC_Wrapper():

    def get_command_line_args(self, runargs, config):
        """Generate command line to call planner from UP."""

        instance = runargs["instance"]

        try:
            import up_ac
            path = '/' + os.path.abspath(up_ac.__file__).strip('/__init__.py')
            path += '/utils'
        except ImportError:
            path = os.getcwd().rsplit('up_ac', 1)[0]
            if path[-1] != "/":
                path += "/"
            path += 'up_ac/utils'

        exc = path + '/sel_call_up.py'

        cmd = f"stdbuf -oL python3 {exc} --i {instance} --c \"{config}\" "
        return cmd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--runargs', type=ast.literal_eval)
    parser.add_argument('--config', type=ast.literal_eval)

    args = vars(parser.parse_args())

    wrapper = UP_AC_Wrapper()
