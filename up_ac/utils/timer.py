from functools import wraps
import multiprocessing
import os
import signal
import time


def parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer


class TimeExceededException(Exception):
    pass


def function_runner(*args, **kwargs):
    """Used as a wrapper function to handle
    returning results on the multiprocessing side"""

    send_end = kwargs.pop("__send_end")
    function = kwargs.pop("__function")
    try:
        result = function(*args, **kwargs)
    except Exception as e:
        send_end.send(e)
        return
    send_end.send(result)


@parametrized
def run_with_timer(func, max_execution_time):
    @wraps(func)
    def wrapper(*args, **kwargs):
        recv_end, send_end = multiprocessing.Pipe(False)
        kwargs["__send_end"] = send_end
        kwargs["__function"] = func
        
        p = multiprocessing.Process(target=function_runner, args=args,
                                    kwargs=kwargs)
        p.start()
        pid = p.pid
        p.join(max_execution_time)
        if p.is_alive():
            p.terminate()
            p.join()
            if p.is_alive():
                os.kill(pid, signal.SIGKILL)
            raise TimeExceededException("Exceeded Execution Time")
        time.sleep(1)
        if p.is_alive():
            p.kill()
            os.kill(pid, signal.SIGKILL)
        result = recv_end.recv()

        if isinstance(result, Exception):
            raise result

        return result

    return wrapper
