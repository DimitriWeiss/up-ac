from unified_planning.exceptions import UPProblemDefinitionError, UPException
import argparse
import ast
import os
import sys
from unified_planning.io import PDDLReader
import unified_planning as up
import timeit
import signal
from contextlib import contextmanager


reader = PDDLReader()
up.shortcuts.get_environment().credits_stream = None


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def sel_call_up(instance, config):
    path = os.getcwd().rsplit('up_ac', 1)[0]
    if path[-1] != "/":
        path += "/"
    up_ac_path = path + 'up_ac'
    sys.path.append(r"{}".format(up_ac_path))
    sys.path.append(r"{}".format(path + 'up_ac/utils'))
    path += 'up_ac/utils'
    setting_path = f'{path}/sel_feedback_setting'
    reader._env.error_used_name = False

    from Selector_interface import SelectorInterface

    with open(f'{setting_path}/feedback_args.txt', 'r') as f:
        feedback_args = ast.literal_eval(f.read())

    selgaci = SelectorInterface()
    selgaci.read_engine_pcs(
        [feedback_args['engine']], f'{up_ac_path}/engine_pcs')

    feedback = planner_feedback(config, instance, reader,
                                selgaci, feedback_args)

    if feedback_args['metric'] == 'runtime':
        if feedback != feedback_args['timelimit'] and feedback is not None:
            print('Tuned plan runtime is:', feedback)
        else:
            print('Fail! Feedback is:', feedback)
    elif feedback_args['metric'] == 'quality':
        print('Tuned plan quality is:', feedback)


def print_feedback(engine, instance, feedback, verbose):
    if verbose:
        print(f'** Feedback of {engine} on instance\n**' +
              f' {instance}\n** is {feedback}\n\n')


def planner_feedback(config, instance, reader, gaci, fa):
    start = timeit.default_timer()
    instance_p = f'{instance}'

    engine = fa['engine']
    metric = fa['metric']
    mode = fa['mode']
    gray_box = fa['gray_box']
    patience = fa['patience']
    train_set = fa['train_set']
    planner_timelimit = fa['timelimit']
    crash_cost = fa['crash_cost']
    verbose = fa['verbose']

    if metric == 'quality':
        timelimit = planner_timelimit - patience
    else:
        timelimit = planner_timelimit

    try:
        if isinstance(train_set, dict) and \
                isinstance(train_set[instance_p], tuple):
            domain = train_set[instance_p][1]
            problem = train_set[instance_p][0]
            pddl_problem = reader.parse_problem(domain, problem)
        else:
            domain_path = instance_p.rsplit('/', 1)[0]
            domain = f'{domain_path}/domain.pddl'
            pddl_problem = reader.parse_problem(f'{domain}',
                                                f'{instance_p}')
        '''
        if engine == 'tamer' or engine == 'pyperplan':
            @concurrent.process(timeout=timelimit)
            def solve(config, metric, engine,
                      mode, pddl_problem):
                feedback = \
                    gaci.run_engine_config(config,
                                           metric, engine,
                                           mode, pddl_problem,
                                           timelimit)

                return feedback
        else:
        '''
        def solve(config, metric, engine,
                  mode, pddl_problem):
            feedback = \
                gaci.run_engine_config(config,
                                       metric, engine,
                                       mode, pddl_problem,
                                       timelimit)

            return feedback

        try:
            if engine == 'tamer' or engine == 'pyperplan':
                def handler(signum, frame):
                    raise TimeoutError("Function timed out!")

                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timelimit)

                try:
                    feedback = solve(config, metric, engine,
                                     mode, pddl_problem)
                except TimeoutError as te:
                    print(te)
                    feedback = timelimit
                finally:
                    signal.alarm(0)

            else:
                with time_limit(timelimit):
                    feedback = solve(config, metric, engine,
                                     mode, pddl_problem)

        except TimeoutException:
            if metric == 'runtime':
                feedback = timelimit
            elif metric == 'quality':
                feedback = crash_cost

    except (AssertionError, NotImplementedError,
            UPProblemDefinitionError, UPException,
            UnicodeDecodeError) as err:
        if verbose:
            print('\n** Error in planning engine!', err)
        if metric == 'runtime':
            feedback = timelimit
        elif metric == 'quality':
            feedback = crash_cost

    if feedback == 'unsolvable':
        if metric == 'runtime':
            feedback = timelimit
        elif metric == 'quality':
            feedback = crash_cost
  
    if feedback is not None:
        # SMAC always minimizes
        if metric == 'quality':
            print_feedback(engine, instance, feedback, verbose)
            return feedback
        # Solving runtime optimization by passing
        # runtime as result, since smac minimizes it
        elif metric == 'runtime':
            if engine in ('tamer', 'pyperplan'):
                feedback = timeit.default_timer() - start
                if feedback > timelimit:
                    feedback = timelimit
                print_feedback(engine, instance, feedback, verbose)
            else:
                if feedback > timelimit:
                    feedback = timelimit
                print_feedback(engine, instance, feedback, verbose)
            return feedback
    else:
        # Penalizing failed runs
        if metric == 'runtime':
            # Penalty is max runtime in runtime scenario
            feedback = timelimit
            print_feedback(engine, instance, feedback, verbose)
        else:
            # Penalty is defined by user in quality scenario
            feedback = crash_cost
            print_feedback(engine, instance, feedback, verbose)

        return feedback


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--i', type=str)
    parser.add_argument('--c', type=str)

    args = vars(parser.parse_args())

    sel_call_up(args['i'], ast.literal_eval(args['c']))
