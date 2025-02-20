"""Testing the UPF library."""

from unified_planning.io import PDDLReader
from unified_planning.shortcuts import *
import unified_planning as up
from types import MethodType
import signal
from contextlib import contextmanager
import time
import os
import up_enhsp


import multiprocessing
from functools import wraps

from typing import IO, Any, Callable, Optional, List, Tuple, Union, cast
from unified_planning.io import PDDLWriter
import tempfile
import subprocess
from unified_planning.engines.results import (
    LogLevel,
    PlanGenerationResult,
    PlanGenerationResultStatus,
)
#from up_fast_downward import utils
from unified_planning.engines.engine import OperationMode
import select


USE_ASYNCIO_ON_UNIX = False
ENV_USE_ASYNCIO = os.environ.get("UP_USE_ASYNCIO_PDDL_PLANNER")
if ENV_USE_ASYNCIO is not None:
    USE_ASYNCIO_ON_UNIX = ENV_USE_ASYNCIO.lower() in ["true", "1"]

from unified_planning.engines import pddl_planner


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


def _solve_enhsp_any(
    self,
    problem: "up.model.AbstractProblem",
    heuristic: Optional[Callable[["up.model.state.State"],
                        Optional[float]]] = None,
    timeout: Optional[float] = None,
    output_stream: Optional[Union[Tuple[IO[str], IO[str]], IO[str]]] = None,
    anytime=False
) -> "up.engines.results.PlanGenerationResult":
    self._mode_running = OperationMode.ANYTIME_PLANNER
    assert isinstance(problem, up.model.Problem)
    self._writer = PDDLWriter(
        problem, self._needs_requirements, self._rewrite_bool_assignments
    )
    plan = None
    print('ANYTIME', anytime)
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
                        run_command_asyncio(
                            self, cmd, output_stream=output_stream, timeout=timeout
                        )
                    )
                else:
                    exec_res = pddl_planner.run_command_posix_select(
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
    plan = None
    if anytime:
        self._mode_running = OperationMode.ANYTIME_PLANNER
    else:
        self._mode_running = OperationMode.ONESHOT_PLANNER
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
        print(cmd)
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
                        run_command_asyncio(
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


def _solve(
    self,
    problem: "up.model.AbstractProblem",
    heuristic: Optional[Callable[["up.model.state.State"], Optional[float]]] = None,
    timeout: Optional[float] = None,
    output_stream: Optional[Union[Tuple[IO[str], IO[str]], IO[str]]] = None,
) -> "up.engines.results.PlanGenerationResult":
    print('IN _SOLVE')
    # print(isinstance(problem, up.model.Problem))
    print(type(problem))
    # assert isinstance(problem, up.model.Problem)
    print('BEFORE WRITER')
    self._writer = PDDLWriter(
        problem, self._needs_requirements, self._rewrite_bool_assignments
    )
    print('AFTER WRITER')
    
    plan = None
    logs: List["up.engines.results.LogMessage"] = []
    with tempfile.TemporaryDirectory() as tempdir:
        domain_filename = os.path.join(tempdir, "domain.pddl")
        problem_filename = os.path.join(tempdir, "problem.pddl")
        plan_filename = os.path.join(tempdir, "plan.txt")
        self._writer.write_domain(domain_filename)
        self._writer.write_problem(problem_filename)
        if self._mode_running == OperationMode.ONESHOT_PLANNER:
            cmd = self._get_cmd(domain_filename, problem_filename, plan_filename)
        elif self._mode_running == OperationMode.ANYTIME_PLANNER:
            assert isinstance(
                self, up.engines.pddl_anytime_planner.PDDLAnytimePlanner
            )
            cmd = self._get_anytime_cmd(
                domain_filename, problem_filename, plan_filename
            )
        if output_stream is None:
            # If we do not have an output stream to write to, we simply call
            # a subprocess and retrieve the final output and error with communicate
            
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            timeout_occurred: bool = False
            proc_out: List[str] = []
            proc_err: List[str] = []
            print(cmd)
            print('Subprocess PID:', process.pid)
            try:
                out_err_bytes = process.communicate(timeout=timeout)
                proc_out, proc_err = [[x.decode()] for x in out_err_bytes]
            except subprocess.TimeoutExpired:
                timeout_occurred = True
                print('Killing', process.pid)
                os.kill(process.pid, signal.SIGKILL)
            retval = process.returncode
        else:
            if sys.platform == "win32":
                # On windows we have to use asyncio (does not work inside notebooks)
                try:
                    loop = asyncio.ProactorEventLoop()
                    exec_res = loop.run_until_complete(
                        run_command_asyncio(
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
                        run_command_asyncio(
                            self, cmd, output_stream=output_stream, timeout=timeout
                        )
                    )
                else:
                    exec_res = run_command_posix_select(
                        self, cmd, output_stream=output_stream, timeout=timeout
                    )
            timeout_occurred, (proc_out, proc_err), retval = exec_res

        print('retval', retval)

        logs.append(up.engines.results.LogMessage(LogLevel.INFO, "".join(proc_out)))
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
                plan=None,
                log_messages=logs,
                engine_name=self.name,
            )
    # print(problem, plan, retval, logs)
    print('1')
    status: PlanGenerationResultStatus = self._result_status(
        problem, plan, retval, logs
    )
    print('2')
    res = PlanGenerationResult(
        status, plan, log_messages=logs, engine_name=self.name
    )
    print('3')
    problem_kind = problem.kind
    if problem_kind.has_continuous_time() or problem_kind.has_discrete_time():
        if isinstance(plan, up.plans.TimeTriggeredPlan) or plan is None:
            return up.engines.results.correct_plan_generation_result(
                res, problem, self._get_engine_epsilon()
            )
    print('4')
    return res

'''
def _solve_ff(
        self,
        problem: "up.model.AbstractProblem",
        heuristic: Optional[Callable[["up.model.state.State"], Optional[float]]] = None,
        timeout: Optional[float] = None,
        output_stream: Optional[Union[Tuple[IO[str], IO[str]], IO[str]]] = None,) -> "up.engines.results.PlanGenerationResult":
    assert isinstance(problem, up.model.Problem)

    # add a new goal atom (initially false) plus an action that has the
    # original goal as precondition and sets the new goal atom
    print('1')
    print(type(problem))
    modified_problem, _, _ = utils.introduce_artificial_goal_action(problem)
    print('2')
    print(type(modified_problem))
    res = _solve_ffat(modified_problem, heuristic, timeout, output_stream)
    return res


def _solve_ffat(
    self,
    problem: "up.model.AbstractProblem",
    heuristic: Optional[Callable[["up.model.state.State"], Optional[float]]] = None,
    timeout: Optional[float] = None,
    output_stream: Optional[Union[Tuple[IO[str], IO[str]], IO[str]]] = None,
    anytime: bool = False,
):
    if anytime:
        self._mode_running = OperationMode.ANYTIME_PLANNER
    else:
        self._mode_running = OperationMode.ONESHOT_PLANNER
    return _solve(problem, heuristic, timeout, output_stream)
'''

#####################################################################################################################################

import subprocess, os, signal


def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    ps_command = subprocess.Popen("ps -o pid --ppid %d --noheaders" % parent_pid, shell=True, stdout=subprocess.PIPE)
    ps_output = ps_command.stdout.read()
    retcode = ps_command.wait()
    assert retcode == 0, "ps command returned %d" % retcode
    for pid_str in ps_output.split("\n")[:-1]:
        os.kill(int(pid_str), sig)


def _get_cmd_enhsp(self, domain_filename: str, problem_filename: str,
                   plan_filename: str):
    path = os.path.abspath(up_enhsp.__file__)
    path = path.rsplit('/', 1)[0]
    command = ['ulimit', '-t', f'{300}', ';', 'timeout',
               '-k', '5', '-s', 'SIGKILL',
               f'{300}s', 'java', '-Xms2g', '-Xmx2g',
               '-XX:+UseSerialGC',
               '-jar', f'{path}/ENHSP/enhsp.jar',
               '-o', domain_filename, '-f', problem_filename,
               '-sp', plan_filename]

    command = map(str.strip, command)
    command = ' '.join(command)

    return command


def _get_anytime_cmd_enhsp(self, domain_filename: str,
                           problem_filename: str,
                           plan_filename: str):
    path = os.path.abspath(up_enhsp.__file__)
    path = path.rsplit('/', 1)[0]
    command = ['ulimit', '-t', f'{300}', ';', 'timeout',
               '-k', '5', '-s', 'SIGKILL',
               f'{300}s', 'java', '-Xms2g', '-Xmx2g',
               '-XX:+UseSerialGC', 
               '-jar', f'{path}/ENHSP/enhsp.jar',
               '-o', domain_filename, '-f', problem_filename,
               '-sp', plan_filename, '-anytime', '-h', 'hadd', '-s', 'gbfs']

    command = map(str.strip, command)
    command = ' '.join(command)

    print('\n\nCOMMAND\n\n', command)

    return command


def _solve_anytime_enhsp(
    self,
    problem: "up.model.AbstractProblem",
    heuristic: Optional[Callable[["up.model.state.State"], Optional[float]]] = None,
    timeout: Optional[float] = None,
    output_stream: Optional[Union[Tuple[IO[str], IO[str]], IO[str]]] = None,
    anytime: bool = False,
):
    if anytime:
        self._mode_running = OperationMode.ANYTIME_PLANNER
    else:
        self._mode_running = OperationMode.ONESHOT_PLANNER
    print('_solve anytime planner')
    print(anytime)
    return _solve_enhsp(problem, heuristic, timeout, output_stream)


def _solve_ff(
    self,
    problem: "up.model.AbstractProblem",
    heuristic: Optional[Callable[["up.model.state.State"], Optional[float]]] = None,
    timeout: Optional[float] = None,
    output_stream: Optional[Union[Tuple[IO[str], IO[str]], IO[str]]] = None,
    anytime: bool = False,
) -> "up.engines.results.PlanGenerationResult":
    print('IN _SOLVE')
    # print(isinstance(problem, up.model.Problem))
    print(type(problem))
    assert isinstance(problem, up.model.Problem)
    problem, _, _ = utils.introduce_artificial_goal_action(problem)
    if anytime:
        self._mode_running = OperationMode.ANYTIME_PLANNER
    else:
        self._mode_running = OperationMode.ONESHOT_PLANNER
    print('BEFORE WRITER')
    self._writer = PDDLWriter(
        problem, self._needs_requirements, self._rewrite_bool_assignments
    )
    print('AFTER WRITER')
    
    plan = None
    logs: List["up.engines.results.LogMessage"] = []
    with tempfile.TemporaryDirectory() as tempdir:
        domain_filename = os.path.join(tempdir, "domain.pddl")
        problem_filename = os.path.join(tempdir, "problem.pddl")
        plan_filename = os.path.join(tempdir, "plan.txt")
        self._writer.write_domain(domain_filename)
        self._writer.write_problem(problem_filename)
        if self._mode_running == OperationMode.ONESHOT_PLANNER:
            cmd = self._get_cmd(domain_filename, problem_filename, plan_filename)
        elif self._mode_running == OperationMode.ANYTIME_PLANNER:
            assert isinstance(
                self, up.engines.pddl_anytime_planner.PDDLAnytimePlanner
            )
            cmd = self._get_anytime_cmd(
                domain_filename, problem_filename, plan_filename
            )

        limits = ['--translate-time-limit', f'{timeout - 10}',
                  '--overall-time-limit', f'{timeout}']
        cmd[6:6] = limits
        if output_stream is None:
            # If we do not have an output stream to write to, we simply call
            # a subprocess and retrieve the final output and error with communicate
            
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            timeout_occurred: bool = False
            proc_out: List[str] = []
            proc_err: List[str] = []
            print(cmd)
            print('Subprocess PID:', process.pid)
            try:
                out_err_bytes = process.communicate(timeout=timeout)
                proc_out, proc_err = [[x.decode()] for x in out_err_bytes]
            except subprocess.TimeoutExpired:
                timeout_occurred = True
                print('Killing', process.pid)
                kill_child_processes(process.pid)
                # os.system('kill -15 -{pid}'.format(pid=-process.pid))
                os.kill(process.pid, signal.SIGKILL)
            retval = process.returncode
        else:
            if sys.platform == "win32":
                # On windows we have to use asyncio (does not work inside notebooks)
                try:
                    loop = asyncio.ProactorEventLoop()
                    exec_res = loop.run_until_complete(
                        run_command_asyncio(
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
                        run_command_asyncio(
                            self, cmd, output_stream=output_stream, timeout=timeout
                        )
                    )
                else:
                    exec_res = run_command_posix_select(
                        self, cmd, output_stream=output_stream, timeout=timeout
                    )
            timeout_occurred, (proc_out, proc_err), retval = exec_res

        print('retval', retval)

        logs.append(up.engines.results.LogMessage(LogLevel.INFO, "".join(proc_out)))
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
                plan=None,
                log_messages=logs,
                engine_name=self.name,
            )
    # print(problem, plan, retval, logs)
    print('1')
    status: PlanGenerationResultStatus = self._result_status(
        problem, plan, retval, logs
    )
    print('2')
    res = PlanGenerationResult(
        status, plan, log_messages=logs, engine_name=self.name
    )
    print('3')
    problem_kind = problem.kind
    if problem_kind.has_continuous_time() or problem_kind.has_discrete_time():
        if isinstance(plan, up.plans.TimeTriggeredPlan) or plan is None:
            return up.engines.results.correct_plan_generation_result(
                res, problem, self._get_engine_epsilon()
            )
    print('4')
    return res


import asyncio


def _solve_symk(
    self,
    problem: "up.model.AbstractProblem",
    heuristic: Optional[Callable[["up.model.state.State"], Optional[float]]] = None,
    timeout: Optional[float] = None,
    output_stream: Optional[Union[Tuple[IO[str], IO[str]], IO[str]]] = None,
    anytime: bool = False,
) -> "up.engines.results.PlanGenerationResult":
    print('IN _SOLVE')
    # print(isinstance(problem, up.model.Problem))
    print(type(problem))
    assert isinstance(problem, up.model.Problem)
    # problem, _, _ = utils.introduce_artificial_goal_action(problem)
    if anytime:
        self._mode_running = OperationMode.ANYTIME_PLANNER
    else:
        self._mode_running = OperationMode.ONESHOT_PLANNER
    print('BEFORE WRITER')
    self._writer = PDDLWriter(
        problem, self._needs_requirements, self._rewrite_bool_assignments
    )
    print('AFTER WRITER')
    
    plan = None
    logs: List["up.engines.results.LogMessage"] = []
    with tempfile.TemporaryDirectory() as tempdir:
        domain_filename = os.path.join(tempdir, "domain.pddl")
        problem_filename = os.path.join(tempdir, "problem.pddl")
        plan_filename = os.path.join(tempdir, "plan.txt")
        self._writer.write_domain(domain_filename)
        self._writer.write_problem(problem_filename)
        if self._mode_running == OperationMode.ONESHOT_PLANNER:
            cmd = self._get_cmd(domain_filename, problem_filename, plan_filename)
        elif self._mode_running == OperationMode.ANYTIME_PLANNER:
            assert isinstance(
                self, up.engines.pddl_anytime_planner.PDDLAnytimePlanner
            )
            cmd = self._get_anytime_cmd(
                domain_filename, problem_filename, plan_filename
            )

        limits = ['--translate-time-limit', f'{timeout - 10}',
                  '--overall-time-limit', f'{timeout}']
        cmd[6:6] = limits
        if output_stream is None:
            # If we do not have an output stream to write to, we simply call
            # a subprocess and retrieve the final output and error with communicate
            
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            timeout_occurred: bool = False
            proc_out: List[str] = []
            proc_err: List[str] = []
            print(cmd)
            print('Subprocess PID:', process.pid)
            try:
                out_err_bytes = process.communicate(timeout=timeout)
                proc_out, proc_err = [[x.decode()] for x in out_err_bytes]
            except subprocess.TimeoutExpired:
                timeout_occurred = True
                print('Killing', process.pid)
                kill_child_processes(process.pid)
                # os.system('kill -15 -{pid}'.format(pid=-process.pid))
                os.kill(process.pid, signal.SIGKILL)
            retval = process.returncode
        else:
            if sys.platform == "win32":
                # On windows we have to use asyncio (does not work inside notebooks)
                try:
                    loop = asyncio.ProactorEventLoop()
                    exec_res = loop.run_until_complete(
                        run_command_asyncio(
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
                        run_command_asyncio(
                            self, cmd, output_stream=output_stream, timeout=timeout
                        )
                    )
                else:
                    exec_res = run_command_posix_select(
                        self, cmd, output_stream=output_stream, timeout=timeout
                    )
            timeout_occurred, (proc_out, proc_err), retval = exec_res

        print('retval', retval)

        logs.append(up.engines.results.LogMessage(LogLevel.INFO, "".join(proc_out)))
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
                plan=None,
                log_messages=logs,
                engine_name=self.name,
            )
    # print(problem, plan, retval, logs)
    print('1')
    status: PlanGenerationResultStatus = self._result_status(
        problem, plan, retval, logs
    )
    print('2')
    res = PlanGenerationResult(
        status, plan, log_messages=logs, engine_name=self.name
    )
    print('3')
    problem_kind = problem.kind
    if problem_kind.has_continuous_time() or problem_kind.has_discrete_time():
        if isinstance(plan, up.plans.TimeTriggeredPlan) or plan is None:
            return up.engines.results.correct_plan_generation_result(
                res, problem, self._get_engine_epsilon()
            )
    print('4')
    return res


import sys
import up_fast_downward
import up_symk
import up_lpg


def _base_cmd_ff(self, plan_filename: str):
    path = os.path.abspath(up_fast_downward.__file__)
    path = path.rsplit('/', 1)[0]
    downward = path + "/downward/fast-downward.py"
    assert sys.executable, "Path to interpreter could not be found"
    cmd = [sys.executable, downward, "--plan-file", plan_filename]
    if self._fd_search_time_limit is not None:
        cmd += ["--search-time-limit", self._fd_search_time_limit]
    cmd += ["--translate-time-limit", "180"]
    cmd += ["--overall-time-limit", "185"]
    cmd += ["--log-level", self._log_level]
    return cmd


def _base_cmd_symk(self, plan_filename: str) -> List[str]:
    path = os.path.abspath(up_symk.__file__)
    path = path.rsplit('/', 1)[0]
    downward = path + "/symk/fast-downward.py"
    assert sys.executable, "Path to interpreter could not be found"
    cmd = [sys.executable, downward, "--plan-file", plan_filename]
    if self._symk_search_time_limit is not None:
        cmd += ["--search-time-limit", self._symk_search_time_limit]
    # Making sure ff really stops
    cmd += ["--translate-time-limit", "70"]
    cmd += ["--overall-time-limit", "90"]
    cmd += ["--log-level", self._log_level]
    return cmd


def _get_cmd(self, domain_filename: str, problem_filename: str, plan_filename: str) -> List[str]:
    path = os.path.abspath(up_lpg.__file__)
    path = path.rsplit('/', 1)[0] + '/lpg'
    base_command = path, '-o', domain_filename, '-f', problem_filename, '-n', '1', '-out', plan_filename, *self.parameter + ['-cputime', '300']
    return base_command


def _get_anytime_cmd(self, domain_filename: str, problem_filename: str, plan_filename: str) -> List[str]:
    path = os.path.abspath(up_lpg.__file__)
    path = path.rsplit('/', 1)[0] + '/lpg'
    base_command = [path, '-o', domain_filename, '-f', problem_filename, '-n', '1', '-out', plan_filename] + self._options
    return base_command

#####################################################################################################################################


def parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer


class TimeExceededException(Exception):
    pass


# PART 1
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
        
        # PART 2
        p = multiprocessing.Process(target=function_runner, args=args, kwargs=kwargs)
        p.start()
        pid = p.pid
        print('PID', pid)
        p.join(max_execution_time)
        if p.is_alive():
            p.terminate()
            p.join()
            if p.is_alive():
                print('KILLING:', pid)
                os.kill(pid, signal.SIGKILL)
            raise TimeExceededException("Exceeded Execution Time")
        time.sleep(3)
        if p.is_alive():
            p.kill()
            print('KILLING:', pid)
            os.kill(pid, signal.SIGKILL)
        result = recv_end.recv()

        if isinstance(result, Exception):
            raise result

        return result

    return wrapper


class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        # signal.alarm(0)
        raise TimeoutException("Timed out!")
        print('\nTime out')
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


instances = []

pddl_reader = PDDLReader()
pddl_reader._env.error_used_name = False
up.shortcuts.get_environment().credits_stream = None

# inst_sets = {'2014_agl': 'agl_inst_domain_pairs_filtered_paper_up_short.txt',
#             '2014_sat': 'sat_inst_domain_pairs_filtered_paper_up_short.txt',
#             '2018_2023': 'inst_domain_pairs_filtered_paper_up_100_local_short.txt'}

inst_sets = {'i': 'all_paper_domains_local.txt'}

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError
from time import sleep
from pebble import concurrent

ff_params = {'fast_downward_search_config': "let(hlm,landmark_sum(lm_factory=lm_reasonable_orders_hps(lm_rhw()),transform=adapt_costs(one),pref=false),let(hff,ff(transform=adapt_costs(one)),lazy_greedy([hff,hlm],preferred=[hff,hlm],cost_type=one,reopen_closed=false)))"}
symk_params = {'symk_search_config': "let(hlm,landmark_sum(lm_factory=lm_reasonable_orders_hps(lm_rhw()),transform=adapt_costs(one),pref=false),let(hff,ff(transform=adapt_costs(one)),lazy_greedy([hff,hlm],preferred=[hff,hlm],cost_type=one,reopen_closed=false)))"}

lpg_params = {'adapt_all_diff': '0', 'adaptfirst': '0', 'avoid_best_action_cycles': '0', 'bestfirst': '0', 'choose_min_numA_fact': '1', 'comp_mutex': '0', 'consider_relaxed_plan_for_inconsistences': '0', 'cputime_localsearch': 1200.0, 'cri_insertion_add_mutex': '0', 'cri_intermediate_levels': '1', 'cri_update_iterations': 0.0, 'criprecond': '6', 'donot_try_suspected_actions': '1', 'evaluate_mutex_for_action_remotion': '0', 'evaluate_threated_supported_preconds_of_neighb_action': '0', 'evaluation_function': '1', 'extended_effects_evaluation': '0', 'extended_unsupported_goals': '0', 'fast_best_action_evaluation': '0', 'force_neighbour_insertion': '0', 'heuristic': '6', 'hpar_cut_neighb': '3', 'ichoice': '25', 'improve_reachability': '0', 'incremental_goal_resolution': '0', 'inst_duplicate_param': '0', 'inst_with_contraddicting_objects': '0', 'lpar_cut_neighb': '1', 'maxnoise': 60.0, 'mutex_and_additive_effects': '1', 'ncut_neighb': '0', 'no_hcut_neighb': '1', 'no_insert_threated_act_in_neighb': '1', 'no_lcut_neighb': '1', 'no_mutex_with_additive_effects': '0', 'no_pruning': '0', 'noise': 0.1, 'nomutex': '1', 'nonuniform_random': '0', 'not_extended_unsupported_facts': '1', 'not_supported_preconds_evaluation': '1', 'notabu_act': '1', 'notabu_fct': '1', 'npar_cut_neighb': 20.0, 'numtry': 500.0, 'onlysearchcostx1stsol': '0', 'penalization_coeff': 2.0, 'penalize_inconsistence': 0.0, 'relaxed_examination': '0', 'relaxed_neighborhood_evaluation': '0', 'remove_act_next_step': '0', 'reset_extended_unsupported_facts': '0', 'ri_list': '0', 'searchcostx1stsol': '0', 'static_noise': '0', 'stop_remove_act': '0', 'tabu_length': 5.0, 'total_time_goal': '0', 'twalkplan': '0', 'verifyAf': '0', 'verifyincchoice': '0', 'verifyinit': '0', 'wcost': 1.0, 'weight_input_plan_cost': 0.0, 'weight_mutex_in_relaxed_plan': 1.0, 'wtime': 0.0, 'zero_num_A': '0', 'lagrange': '0', 'numrestart': 9.0, 'lm_decrme': 5e-07, 'lm_decrprec': 5e-07, 'lm_incrme': 0.001, 'lm_incrprec': 0.001, 'lm_multilevel': '0'}

# with AnytimePlanner(name='lpg-anytime') as planner:  # OneshotPlanner(name=p, params={}) as planner:
#    print('PLANNER', planner)

for name, iss in inst_sets.items():

    with open(f'{iss}', 'r') as f:
        for line in f:
            instances.append(eval(line.strip()))

    worked_for = {}

    # planners = ['fast-downward', 'lpg', 'enhsp', 'tamer', 'pyperplan']

    # anytime_planners = ['fast-downward', 'lpg-anytime', 'enhsp', 'tamer', 'pyperplan']

    planners = ['symk']

    # timeouts = [{'fast_downward_search_time_limit': '10'}, {'-cputime': 10}, {}, {}, {}]

    timeouts = [{}, {}, {}]

    # instances = instances[80:]

    def _manage_parameters_n(self, command):
        if self.params is not None:
            command += self.params.split()
            command += '-timeout'
            command += '-10'
        else:
            command += ['-h', 'hadd', '-s', 'gbfs', '-timeout', '90']
        return command

    @run_with_timer(max_execution_time=300)
    def run_stuff(p, param, problem):
        @concurrent.process(timeout=300, daemon=False)
        def run_planner(p, param, problem):
            print('run stuff')
            with AnytimePlanner(name=p, params={'symk_anytime_search_config': 'astar(blind())'}) as planner:
                
                # with OneshotPlanner(name=p, params={'symk_search_config': 'astar(blind())'}) as planner:
                print('PLANNER', planner)
                print(planner.name)
                if p == 'enhsp':
                    planner._get_cmd = MethodType(_get_cmd_enhsp, planner)
                    planner._solve = MethodType(_solve_enhsp, planner)
                elif p == 'enhsp-any':
                    print('!!!!!!!!!!!!!!')
                    
                    planner._solve = MethodType(_solve_enhsp, planner)
                    planner._get_anytime_cmd = MethodType(_get_anytime_cmd_enhsp, planner)
                    # planner.run_command_posix_select = MethodType(run_command_posix_select_enhsp, planner)
                    # planner._manage_parameters = MethodType(_manage_parameters_n, planner)
                elif p == 'lpg':
                    if planner.name == 'lpg':
                        planner._get_cmd = MethodType(_get_cmd, planner)
                elif p == 'lpg-anytime':
                    if planner.name == 'lpg-anytime':
                        planner._get_anytime_cmd = MethodType(_get_anytime_cmd, planner)
                    # planner._solve = MethodType(_solve, planner)
                elif p == 'fast-downward':
                    planner._base_cmd = MethodType(_base_cmd_ff, planner)
                    # planner._solve = MethodType(_solve_ff, planner)
                elif p == 'symk':
                    planner._base_cmd = MethodType(_base_cmd_symk, planner)
                # elif p == 'fast-downward':
                #    _solve = _solve_ff
                #    planner._solve = MethodType(_solve, planner)
                #    print('WE ARE HERE')
                print('\n\nStarting', problem.name)
                # with ThreadPoolExecutor() as tpe:
                #    future = tpe.submit(planner.solve, problem)
                #    try:
                #        result = future.exception(timeout=30)
                #    except TimeoutError:
                #        print('TIMEOUT')
                '''
                if p == 'pyperplan' or p == 'tamer':
                    result = planner.solve(problem)
                else:
                    print('here')
                    result = planner.solve(problem, timeout=300)
                    print('yes')
                '''
                
                result = None
                for res in planner.get_solutions(problem, timeout=180):
                    print(res.status)
                    if res is not None and (res.status == up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING or res.status == up.engines.PlanGenerationResultStatus.SOLVED_OPTIMALLY or res.status == up.engines.PlanGenerationResultStatus.INTERMEDIATE):
                        result = res
                        continue 

                    # result = result[0]
                    print('Result', result)
                    print('Result Status:', result.status)
                    # print(result)
                    res = result
                    if result is not None and \
                        (result.status in
                            (up.engines.
                                PlanGenerationResultStatus.SOLVED_SATISFICING,
                                up.engines.
                                PlanGenerationResultStatus.SOLVED_OPTIMALLY)):
                        print(result.status)
                        break
                        # print(f"{p} returned: %s" % result.plan) .log_messages[0].message
                        '''
                        print(problem.name)
                        print('\n\nRESULT')
                        print(result)
                        print('RESULT END\n\n')
                        print('\n\nRESULT Log_messages')
                        print('result.metrics:', result.metrics)
                        print('______________________')
                        print(dir(result))
                        print('______________________')
                        print(result.log_messages)
                        print('______________________')
                        print(result.log_messages[0])
                        print('______________________')
                        print(result.log_messages[0].message)
                        print('RESULT Log_messages END\n\n')
                        for m in result.log_messages[0].message.split('\n'):
                            print('m:', m)
                            if 'Metric' in m:
                                print(m)
                        plan_cost = next((m for m in result.log_messages[0].message.split('\n') if 'Plan cost' in m), None)
                        print('Quality:', plan_cost.split(' ')[5])
                        '''
                        
                    else:
                        print("No plan found.")
        
        run_planner(p, param, problem)

    for i, p in enumerate(planners):
        worked_for[p] = []

        for inst in instances:

            # if inst[0] != '/media/dweiss/Transcend7/AIPlan4EU/Instances/IPC2014/TESTING/seq-agl/CaveDiving/testing01.pddl':
            #    continue

            pyperplan_domains = ['Barman', 'Childsnack', 'Floortile', 'Parking', 'Thoughtful', 'Visitall']

            if any(ext in inst[0] for ext in pyperplan_domains):
                pass
            else:
                continue

            print('\n\n______________________________________________')
            print(inst[1])
            print(inst[0])
            
            # if 'Hiking' not in inst[0] and 'Floortile' not in inst[0] and 'CityCar' not in inst[0]:
            #    continue
            # if p == 'tamer' and 'termes' in inst[0]:
            #    continue
            problem = pddl_reader.parse_problem(inst[1], inst[0])
            with time_limit(300):
                # with TimeoutAfter(timeout=30, exception=TimeoutError):
                run_stuff(p, timeouts[i], problem)
            worked_for[p].append(inst)
            print('WORKED!!!!!!!!!!!!!\n', problem.name, '\n\n')
            try:
                problem = pddl_reader.parse_problem(inst[1], inst[0])
                with time_limit(300):
                    # with TimeoutAfter(timeout=30, exception=TimeoutError):
                    run_stuff(p, timeouts[i], problem)
                worked_for[p].append(inst)
                print('WORKED!!!!!!!!!!!!!\n', problem.name, '\n\n')
            except Exception as e:
                if isinstance(e, TimeoutException):
                    worked_for[p].append(inst)
                    print('Exception:', e)
                else:
                    print('Another Exception')
                    print('Exception:', e)
            finally:
                pass
            print('-------------------------------------------------')
            time.sleep(1)

        # with open(f'instances_further_filtered{name}', 'w') as f:
        #    for wf in worked_for[p]:
        #        f.write(f"{wf}\n")
    time.sleep(3)
