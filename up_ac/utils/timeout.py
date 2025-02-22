"""Functions to enforce timeout in cmd call."""

import subprocess
import sys
import tempfile
import os
import signal
import multiprocessing
from functools import wraps
import time
import unified_planning as up
from unified_planning.engines.engine import OperationMode
from unified_planning.engines.results import (
    LogLevel,
    PlanGenerationResult,
    PlanGenerationResultStatus,
)
from unified_planning.io import PDDLWriter
from typing import IO, Callable, Optional, List, Tuple, Union, Any, cast

from types import MethodType
import up_enhsp
import up_fast_downward
import up_symk
import warnings
import pytamer
import up_lpg
from up_tamer.engine import TState
from unified_planning.engines import pddl_planner
import select
import asyncio
from pebble import concurrent


USE_ASYNCIO_ON_UNIX = False
ENV_USE_ASYNCIO = os.environ.get("UP_USE_ASYNCIO_PDDL_PLANNER")
if ENV_USE_ASYNCIO is not None:
    USE_ASYNCIO_ON_UNIX = ENV_USE_ASYNCIO.lower() in ["true", "1"]


def linux_timeout(planner, engine, timelimit, config):
    """Adjust planner cal for AC on Linux.

    parameter engine: str, name of engine.
    parameter timelimit: int, seconds to run for.
    """
    if engine in ('enhsp', 'enhsp-any'):
        path = os.path.abspath(up_enhsp.__file__)
        path = path.rsplit('/', 1)[0]

        # print('config in to', config)
        config = config['params'].split()

        #search = config['search_algorithm']
        #heuristic = config['heuristic']

        if planner.name == 'enhsp':
            def _get_cmd(self, domain_filename: str, problem_filename: str,
                         plan_filename: str):
                base_command = ['ulimit', '-t', f'{timelimit + 5}', ';',
                                'timeout', '-k', '5', '-s', 'SIGKILL',
                                f'{timelimit}s', 'java', '-Xms2g', '-Xmx2g',
                                '-XX:+UseSerialGC', '-jar',
                                f'{path}/ENHSP/enhsp.jar',
                                '-o', domain_filename, '-f', problem_filename,
                                '-sp', plan_filename]

                base_command = self._manage_parameters(base_command)

                base_command = map(str.strip, base_command)
                base_command = ' '.join(base_command)

                #print('base_command', base_command)

                return base_command

            planner._get_cmd = MethodType(_get_cmd, planner)
            planner._solve = MethodType(_solve_enhsp, planner)

        elif planner.name == 'Anytime-enhsp':
            def _get_anytime_cmd(self, domain_filename: str,
                                 problem_filename: str,
                                 plan_filename: str):
                command = ['ulimit', '-t', f'{timelimit + 5}', ';', 'timeout',
                           '-k', '5', '-s', 'SIGKILL',
                           f'{timelimit}s', 'java', '-Xms2g', '-Xmx2g',
                           '-XX:+UseSerialGC', 
                           '-jar', f'{path}/ENHSP/enhsp.jar',
                           '-o', domain_filename, '-f', problem_filename,
                           '-sp', plan_filename, '-s', f'{config[3]}',
                           '-h', f'{config[1]}', '-anytime']

                command = map(str.strip, command)
                command = ' '.join(command)

                return command

            planner._solve = MethodType(_solve_enhsp, planner)
            planner._get_anytime_cmd = MethodType(_get_anytime_cmd, planner)

        elif planner.name == 'SAT-enhsp':
            def _get_cmd(self, domain_filename: str, problem_filename: str,
                         plan_filename: str):
                command = ['ulimit', '-t', f'{timelimit + 5}', ';', 'timeout',
                           '-k', '5', '-s', 'SIGKILL',
                           f'{timelimit}s', 'java', '-Xms2g', '-Xmx2g',
                           '-XX:+UseSerialGC',
                           '-jar', f'{path}/ENHSP/enhsp.jar',
                           '-o', domain_filename, '-f', problem_filename,
                           '-sp', plan_filename, '-s', f'{config[3]}',
                           '-h', f'{config[1]}']

                command = map(str.strip, command)
                command = ' '.join(command)

                return command

            planner._get_cmd = MethodType(_get_cmd, planner)
            planner._solve = MethodType(_solve_enhsp, planner)

        elif planner.name == 'OPT-enhsp':
            def _get_cmd(self, domain_filename: str, problem_filename: str,
                         plan_filename: str):
                command = ['ulimit', '-t', f'{timelimit + 5}', ';', 'timeout',
                           '-k', '5', '-s', 'SIGKILL',
                           f'{timelimit}s', 'java', '-Xms2g', '-Xmx2g',
                           '-XX:+UseSerialGC', 
                           '-jar', f'{path}/ENHSP/enhsp.jar',
                           '-o', domain_filename, '-f', problem_filename,
                           '-sp', plan_filename, '-s', f'{config[3]}',
                           '-h', f'{config[1]}']

                command = map(str.strip, command)
                command = ' '.join(command)

                return command

            planner._get_cmd = MethodType(_get_cmd, planner)
            planner._solve = MethodType(_solve_enhsp, planner)

    elif engine == 'lpg':
        def _get_cmd(self, domain_filename: str, problem_filename: str,
                     plan_filename: str) -> List[str]:
            path = os.path.abspath(up_lpg.__file__)
            path = path.rsplit('/', 1)[0] + '/lpg'
            base_command = 'timeout', '-k', '5', '-s', 'SIGKILL', \
                f'{timelimit}s', path, '-o', domain_filename, '-f', \
                problem_filename, '-n', '1', '-out', plan_filename, \
                *self.parameter + ['-cputime', f'{timelimit}']
            return base_command

        planner._get_cmd = MethodType(_get_cmd, planner)

    elif engine == 'lpg-anytime':
        def _get_anytime_cmd(self, domain_filename: str,
                             problem_filename: str,
                             plan_filename: str) -> List[str]:
            path = os.path.abspath(up_lpg.__file__)
            path = path.rsplit('/', 1)[0] + '/lpg'
            base_command = ['timeout', '-k', '5', '-s', 'SIGKILL', 
                            f'{timelimit}s', path, '-o', domain_filename, '-f',
                            problem_filename, '-n', '1', '-out', plan_filename,
                            *self.parameter] + self._options
            return base_command

        planner._get_anytime_cmd = MethodType(_get_anytime_cmd, planner)

    elif engine == 'fast-downward':

        def _base_cmd(self, plan_filename: str):
            path = os.path.abspath(up_fast_downward.__file__)
            path = path.rsplit('/', 1)[0]
            downward = path + "/downward/fast-downward.py"
            assert sys.executable, "Path to interpreter could not be found"
            cmd = [sys.executable, downward, "--plan-file", plan_filename]
            if self._fd_search_time_limit is not None:
                cmd += ["--search-time-limit", self._fd_search_time_limit]
            # Making sure ff really stops
            cmd += ["--translate-time-limit",
                    f'{timelimit - int(0.1 * timelimit)}']
            cmd += ["--overall-time-limit", f'{timelimit}']
            cmd += ["--log-level", self._log_level]
            return cmd

        planner._base_cmd = MethodType(_base_cmd, planner)

    elif engine == 'symk':

        def _base_cmd(self, plan_filename: str) -> List[str]:
            path = os.path.abspath(up_symk.__file__)
            path = path.rsplit('/', 1)[0]
            downward = path + "/symk/fast-downward.py"
            assert sys.executable, "Path to interpreter could not be found"
            cmd = [sys.executable, downward, "--plan-file", plan_filename]
            if self._symk_search_time_limit is not None:
                cmd += ["--search-time-limit", self._symk_search_time_limit]
            # Making sure ff really stops
            cmd += ["--translate-time-limit",
                    f'{timelimit - int(0.1 * timelimit)}']
            cmd += ["--overall-time-limit", f'{timelimit}']
            cmd += ["--log-level", self._log_level]
            return cmd

        planner._base_cmd = MethodType(_base_cmd, planner)

    return planner


def _solve_enhsp(
    self,
    problem: "up.model.AbstractProblem",
    heuristic: Optional[Callable[["up.model.state.State"],
                        Optional[float]]] = None,
    timeout: Optional[float] = None,
    output_stream: Optional[Union[Tuple[IO[str], IO[str]], IO[str]]] = None,
    anytime=False
) -> "up.engines.results.PlanGenerationResult":
    assert isinstance(problem, up.model.Problem)
    self._writer = PDDLWriter(
        problem, self._needs_requirements, self._rewrite_bool_assignments
    )
    if anytime:
        self._mode_running = OperationMode.ANYTIME_PLANNER
    else:
        self._mode_running = OperationMode.ONESHOT_PLANNER
    plan = None
    logs: List["up.engines.results.LogMessage"] = []
    with tempfile.TemporaryDirectory() as tempdir:
        domain_filename = os.path.join(tempdir, "domain.pddl")
        problem_filename = os.path.join(tempdir, "problem.pddl")
        plan_filename = os.path.join(tempdir, "plan.txt")
        self._writer.write_domain(domain_filename)
        self._writer.write_problem(problem_filename)
        if self._mode_running == OperationMode.ONESHOT_PLANNER:
            cmd = self._get_cmd(domain_filename, problem_filename,
                                plan_filename)
        elif self._mode_running == OperationMode.ANYTIME_PLANNER:
            assert isinstance(
                self, up.engines.pddl_anytime_planner.PDDLAnytimePlanner
            )
            cmd = self._get_anytime_cmd(
                domain_filename, problem_filename, plan_filename
            )
        if output_stream is None:
            # If we do not have an output stream to write to, we simply call
            # a subprocess and retrieve the final output and error with
            # communicate
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
            )
            timeout_occurred: bool = False
            proc_out: List[str] = []
            proc_err: List[str] = []
            try:
                out_err_bytes = process.communicate(timeout=timeout)
                proc_out, proc_err = [[x.decode()] for x in out_err_bytes]
            except subprocess.TimeoutExpired:
                timeout_occurred = True
            retval = process.returncode
        else:
            if sys.platform == "win32":
                # On windows we have to use asyncio (does not work inside notebooks)
                try:
                    loop = asyncio.ProactorEventLoop()
                    exec_res = loop.run_until_complete(
                        pddl_planner.run_command_asyncio(
                            self, cmd, output_stream=output_stream, timeout=timeout
                        )
                    )
                finally:
                    loop.close()
            else:
                # On non-windows OSs, we can choose between asyncio and posix
                # select (see comment on USE_ASYNCIO_ON_UNIX variable for details)
                if USE_ASYNCIO_ON_UNIX:
                    exec_res = asyncio.run(
                        pddl_planner.run_command_asyncio(
                            self, cmd, output_stream=output_stream, timeout=timeout
                        )
                    )
                else:
                    exec_res = run_command_posix_select_enhsp(
                        self, cmd, output_stream=output_stream, timeout=timeout
                    )
            timeout_occurred, (proc_out, proc_err), retval = exec_res

        logs.append(
            up.engines.results.LogMessage(LogLevel.INFO, "".join(proc_out)))
        logs.append(
            up.engines.results.LogMessage(LogLevel.ERROR, "".join(proc_err))
        )
        if os.path.isfile(plan_filename):
            plan = self._plan_from_file(
                problem, plan_filename, self._writer.get_item_named
            )
        if timeout_occurred and retval != 0:
            return PlanGenerationResult(
                PlanGenerationResultStatus.TIMEOUT,
                plan=plan,
                log_messages=logs,
                engine_name=self.name,
            )
    status: PlanGenerationResultStatus = self._result_status(
        problem, plan, retval, logs
    )
    res = PlanGenerationResult(
        status, plan, log_messages=logs, engine_name=self.name
    )
    problem_kind = problem.kind
    if problem_kind.has_continuous_time() or problem_kind.has_discrete_time():
        if isinstance(plan, up.plans.TimeTriggeredPlan) or plan is None:
            return up.engines.results.correct_plan_generation_result(
                res, problem, self._get_engine_epsilon()
            )
    return res


def run_command_posix_select_enhsp(
    engine: pddl_planner.PDDLPlanner,
    cmd: List[str],
    output_stream: Union[Tuple[IO[str], IO[str]], IO[str]],
    timeout: Optional[float] = None,
) -> Tuple[bool, Tuple[List[str], List[str]], int]:
    """
    Executed the specified command line using posix select, imposing the specified timeout and printing online the output on output_stream.
    The function returns a boolean flag telling if a timeout occurred, a pair of string lists containing the captured standard output and standard error and the return code of the command as an integer

    WARNING: this does not work under Windows because the select function only support sockets and not pipes
    WARNING: The resolution of the timeout parameter is ~ 1 second if output_stream is specified
    """
    proc_out: List[str] = []
    proc_err: List[str] = []
    proc_out_buff: List[str] = []
    proc_err_buff: List[str] = []

    #print(cmd)

    engine._process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    
    timeout_occurred: bool = False
    start_time = time.time()
    last_red_out, last_red_err = 0, 0  # Variables needed for the correct loop exit
    readable_streams: List[Any] = []
    # Exit loop condition: Both stream have nothing left to read or the planner is out of time
    while not timeout_occurred and (
        len(readable_streams) != 2 or last_red_out != 0 or last_red_err != 0
    ):
        readable_streams, _, _ = select.select(
            [engine._process.stdout, engine._process.stderr], [], [], 1.0
        )  # 1.0 is the timeout resolution
        if (
            timeout is not None and time.time() - start_time >= timeout
        ):  # Check if the planner is out of time.
            try:
                engine._process.kill()
            except OSError:
                pass  # This can happen if the process is already terminated
            timeout_occurred = True
        for readable_stream in readable_streams:
            out_in_bytes = readable_stream.readline()
            out_str = out_in_bytes.decode().replace("\r\n", "\n")
            if readable_stream == engine._process.stdout:
                if type(output_stream) is tuple:
                    assert len(output_stream) == 2
                    if output_stream[0] is not None:
                        output_stream[0].write(out_str)
                else:
                    cast(IO[str], output_stream).write(out_str)
                last_red_out = len(out_in_bytes)
                buff = proc_out_buff
                lst = proc_out
            else:
                if type(output_stream) is tuple:
                    assert len(output_stream) == 2
                    if output_stream[1] is not None:
                        output_stream[1].write(out_str)
                else:
                    cast(IO[str], output_stream).write(out_str)
                last_red_err = len(out_in_bytes)
                buff = proc_err_buff
                lst = proc_err
            buff.append(out_str)
            if "\n" in out_str:
                lines = "".join(buff).split("\n")
                for x in lines[:-1]:
                    lst.append(x + "\n")

                buff.clear()
                if lines[-1]:
                    buff.append(lines[-1])
        lastout = "".join(proc_out_buff)
        if lastout:
            proc_out.append(lastout + "\n")
        lasterr = "".join(proc_err_buff)
        if lasterr:
            proc_err.append(lasterr + "\n")
    engine._process.wait()
    return timeout_occurred, (proc_out, proc_err), cast(int, engine._process.returncode)
