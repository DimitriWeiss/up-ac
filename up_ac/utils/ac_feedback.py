"""Functions to get feedback from engines performance."""


def qaul_feedback(engine, result):
    """Transform/parse specific solution quality engine output.

    parameter engine: str, name of engine.
    parameter result: object, planning result.
    """
    feedback = None

    if engine in ('lpg', 'lpg-anytime'):
        feedback = \
            float(
                next((m for m in result.log_messages[0].message.split('\n')
                     if 'Plan quality' in m), None).split(' ')[-2])

    elif engine in ('fast-downward', 'symk'):
        feedback = \
            float(
                next((m for m in result.log_messages[0].message.split('\n')
                     if 'Plan cost' in m), None).split(' ')[5])

    elif engine in ('enhsp', 'enhsp-any'):
        feedback = \
            float(
                next((m for m in result.log_messages[0].message.split('\n')
                     if 'Metric' in m), None).split(':')[1])

    return feedback


def runtime_feedback(engine, result):
    """Transform/parse specific solution runtime engine output.

    parameter engine: str, name of engine.
    parameter result: object, planning result.
    """
    feedback = None

    if engine == 'lpg':
        output = result.log_messages[0].message.split('\n')
        for line in output:
            if 'Total time:' in line:
                feedback = float(line.split(' ')[-1])

    elif engine in ('fast-downward', 'symk'):
        output = result.log_messages[0].message.split('\n')
        for line in output:
            if 'Planner time:' in line:
                feedback = float(line.split(' ')[-1][:-1])

    elif engine == 'enhsp':
        output = result.log_messages[0].message.split('\n')
        for line in output:
            if 'Planning Time' in line:
                feedback = (float(line.split(':')[1])) / 1000  # msec

    elif engine in ('pyperplan', 'tamer'):
        feedback = 'measure'

    return feedback


def gray_box_feedback(engine, result):
    """Transform/parse intermediate output.

    parameter engine: str, name of engine.
    parameter result: object, planning result.
    """
    # TODO
    if engine in ('lpg', 'lpg-anytime'):
        feedback = None

    elif engine in ('fast-downward', 'symk'):
        feedback = None

    return feedback
