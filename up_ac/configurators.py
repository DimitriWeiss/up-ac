"""Functionalities for managing and calling configurators."""
from unified_planning.io import PDDLReader
from unified_planning.exceptions import UPProblemDefinitionError

import json
import timeit
import time
import os
import sys
import subprocess
import psutil
import signal
import multiprocessing

# make sure test can be run from anywhere
path = os.getcwd().rsplit('up-ac', 1)[0]
if path[-1] != "/":
    path += "/"
path += 'up-ac/up_ac'
if not os.path.isfile(sys.path[0] + '/configurators.py') and \
        'up-ac' in sys.path[0]:
    sys.path.append(sys.path[0].rsplit('up-ac', 1)[0] + 'up-ac')
    sys.path.append(sys.path[0].rsplit('up-ac', 1)[0] + 'up-ac/up_ac')
    sys.path.append(sys.path[0].rsplit('up-ac', 1)[0] + 'up-ac/utils')

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from AC_interface import *


class Configurator():
    """Configurator functions."""

    def __init__(self):
        """Initialize generic interface."""
        self.capabilities = {'quality': {
                             'OneshotPlanner': 
                             ['lpg', 'fast-downward', 'enhsp', 'symk',
                              'pyperplan', 'tamer'],
                             'AnytimePlanner':
                             ['fast-downward', 'symk', 'enhsp-any',
                              'lpg-anytime']},
                             'runtime': {
                             'OneshotPlanner':
                             ['lpg', 'fast-downward', 'enhsp', 'symk',
                              'tamer', 'pyperplan']}
                             }
        self.incumbent = None
        self.instance_features = {}
        self.train_set = {}
        self.test_set = {}
        self.reader = PDDLReader()
        self.reader._env.error_used_name = False
        self.metric = None
        self.crash_cost = 0
        self.ac = None
        self.verbose = True

        # Control for integrated planner versions
        '''
        try:
            import up_enhsp
            cai('up-enhsp', '0.0.25')
        except (ImportError, ModuleNotFoundError):
            pass
        try:
            import up_fast_downward
            cai('up-fast-downward', '0.4.1')
        except (ImportError, ModuleNotFoundError):
            pass
        try:
            import up_lpg
            cai('up-lpg', '0.0.11')
        except (ImportError, ModuleNotFoundError):
            pass
        try:
            import up_pyperplan
            cai('up-pyperplan', '1.1.0')
        except (ImportError, ModuleNotFoundError):
            pass
        try:
            import up_symk
            cai('up-symk', '1.3.1')
        except (ImportError, ModuleNotFoundError):
            pass
        try:
            import up_tamer
            cai('up-tamer', '1.1.3')
        except (ImportError, ModuleNotFoundError):
            pass
        '''

    def print_feedback(self, engine, instance, feedback):
        """
        Print feedback from the engine.

        :param engine: Name of the engine.
        :type engine: str
        :param instance: Name of the instance.
        :type instance: str
        :param feedback: Feedback from the engine.
        """
        if self.verbose:
            print(f'** Feedback of {engine} on instance\n**' +
                  f' {instance}\n** is {feedback}\n\n')

    def get_instance_features(self, instance_features=None):
        """
        Save instance features.

        :param instance_features: Instance names and their features in lists.
        :type instance_features: dict, optional
        """
        self.instance_features = instance_features
        if self.verbose:
            print('\nSetting instance features.\n')

    def set_training_instance_set(self, train_set):
        """
        Save training instance set.

        :param train_set: List of instance paths.
        :type train_set: list
        """
        self.train_set = train_set
        if self.verbose:
            print('\nSetting training instance set.\n')

    def set_test_instance_set(self, test_set):
        """
        Set test instance set.

        :param test_set: List of instance paths.
        :type test_set: list
        """
        self.test_set = test_set
        if self.verbose:
            print('\nSetting testing instance set.\n')

    def get_feedback_function(self, gaci, engine, metric, mode,
                              gray_box=False):
        """
        Generate the function to run the engine and get feedback.

        :param gaci: Algorithm Configuration interface object.
        :type gaci: ACInterface
        :param engine: Engine name.
        :type engine: str
        :param metric: Metric, either 'runtime' or 'quality'.
        :type metric: str
        :param mode: Type of planning.
        :type mode: str
        :param gray_box: True if gray box to be used, optional.
        :type gray_box: bool, optional

        :return: Planner feedback function or None if not supported.
        :rtype: function or None
        """
        if engine in self.capabilities[metric][mode]:
            self.metric = metric

            planner_feedback = None

            return planner_feedback
        else:
            if self.verbose:
                print(f'Algorithm Configuration for {metric} of {engine}' + \
                      f' in {mode} is not supported.')
            return None

    def set_scenario(self, engine, param_space, gaci,
                     configuration_time=120, n_trials=400, min_budget=1,
                     max_budget=3, crash_cost=0, planner_timelimit=30,
                     n_workers=1, instances=[], instance_features=None,
                     metric='runtime', popSize=128, evalLimit=2147483647):
        """
        Set up algorithm configuration scenario.

        :param engine: Engine name.
        :type engine: str
        :param param_space: ConfigSpace object.
        :type param_space: ConfigSpace
        :param gaci: AC interface object.
        :type gaci: ACInterface
        :param configuration_time: Overall configuration time budget, optional.
        :type configuration_time: int, optional
        :param n_trials: Maximum number of engine evaluations, optional.
        :type n_trials: int, optional
        :param min_budget: Minimum number of instances to use, optional.
        :type min_budget: int, optional
        :param max_budget: Maximum number of instances to use, optional.
        :type max_budget: int, optional
        :param crash_cost: Cost to use if the engine fails, optional.
        :type crash_cost: int, optional
        :param planner_timelimit: Maximum runtime per evaluation, optional.
        :type planner_timelimit: int, optional
        :param n_workers: Number of cores to utilize, optional.
        :type n_workers: int, optional
        :param instances: Problem instance paths, optional.
        :type instances: list, optional
        :param instance_features: Instance names and lists of features, optional.
        :type instance_features: dict, optional
        :param metric: Optimization metric, optional.
        :type metric: str, optional
        :param popSize: Population size of configs per generation (OAT), optional.
        :type popSize: int, optional
        :param evalLimit: Maximum number of evaluations (OAT), optional.
        :type evalLimit: int, optional
        """
        self.planner_timelimit = planner_timelimit

        scenario = None

        self.scenario = scenario

    def optimize(self, feedback_function=None, gray_box=False):
        """
        Run the algorithm configuration.

        :param feedback_function: Function to run engine and get feedback, optional.
        :type feedback_function: function, optional
        :param gray_box: True if gray box usage, optional.
        :type gray_box: bool, optional

        :return: The best configuration found during optimization.
        :rtype: dict
        """
        if feedback_function is not None:

            return self.incumbent

    def get_process_by_name(self, names):
        """
        Finds all processes matching given names.

        :param names: List of names of processes to get.
        :type names: str

        :return: Processes that match the names.
        :rtype: list of psuitl.Process
        """
        matching_processes = []
        for proc in psutil.process_iter(attrs=['pid', 'name']):
            try:
                if any(proc.info['name'] in names for name in names):
                    matching_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied,
                    psutil.ZombieProcess):
                continue
        return matching_processes

    def trigger_gc_for_children(self, pid):
        """
        Triggers garbage collection for child processes.

        :param pid: PID of parent Process.
        :type names: int
        """
        java_gcc_processes = self.get_process_by_name(["java", "gcc", "g++"])
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            try:
                process_name = child.name().lower()
                pid = child.pid

                if "java" in process_name:
                    print(f"Triggering GC for Java process PID: {pid}",
                          flush=True)
                    subprocess.run(["jcmd", str(pid), "GC.run"])

                elif "gcc" in process_name or "g++" in process_name:
                    print(f"Triggering GC for GCC process PID: {pid}",
                          flush=True)
                    # Sending SIGHUP as a best-effort attempt
                    child.send_signal(signal.SIGHUP)

            except Exception as e:
                print(f"Error triggering GC for process PID {pid}: {e}",
                      flush=True)

        # Monitor memory usage and wait until it reduces
        while any(proc.is_running() for proc in java_gcc_processes):
            try:
                while any(proc.memory_info().rss > 500 * 1024 * 1024 
                          for proc in java_gcc_processes):
                    print("Memory still high, waiting...")
                    time.sleep(5)
            except Exception as e:
                print(e)
                pass

    def kill_process_tree(self, pid, wt):
        """
        Terminate/kill all child processes.

        :param pid: PID of parent Process.
        :type names: int
        :param wt: Time to wait for process to terminate.
        :type names: int
        """
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                child.terminate()
            parent.terminate()

            # Wait for termination
            _, still_alive = psutil.wait_procs(children, timeout=wt)
            for p in still_alive:
                p.kill()  # Force kill if still alive

            parent.wait(wt)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired) as err:
            print(err)

    def evaluate(self, metric, engine, mode, incumbent, gaci,
                 planner_timelimit=300, crash_cost=10000, instances=[]):
        """
        Evaluate performance of found configuration on training set.

        :param metric: Optimization metric.
        :type metric: str
        :param engine: Engine name.
        :type engine: str
        :param mode: Planning mode.
        :type mode: str
        :param incumbent: Parameter configuration to evaluate.
        :type incumbent: dict
        :param gaci: AC interface object.
        :type gaci: ACInterface
        :param planner_timelimit: Max runtime per evaluation, optional.
        :type planner_timelimit: int, optional
        :param crash_cost: Cost if engine fails, optional.
        :type crash_cost: int, optional
        :param instances: Instance paths, optional.
        :type instances: list, optional

        :return: Average performance on the instances.
        :rtype: float
        """
        if incumbent is not None:
            underline = '____________________'

            if not instances:
                instances = self.test_set
            nr_inst = len(instances)
            avg_f = 0

            def solve(incumbent, metric, engine,
                      mode, pddl_problem, conn):
                f = \
                    gaci.run_engine_config(incumbent,
                                           metric, engine,
                                           mode, pddl_problem,
                                           timelimit=planner_timelimit)

                conn.send(f)  # Send result via pipe
                conn.close()

                return f

            for inst in instances:
                if metric == 'runtime':
                    start = timeit.default_timer()

                try:
                    if isinstance(inst, tuple):
                        instance_p = f'{inst[0]}'
                        domain = f'{inst[1]}'
                    else:
                        instance_p = f'{inst}'
                        domain_path = instance_p.rsplit('/', 1)[0]
                        domain = f'{domain_path}/domain.pddl'
                    pddl_problem = self.reader.parse_problem(f'{domain}',
                                                             f'{instance_p}')

                    print('\n\nEvaluatng instance:', instance_p)

                    parent_conn, child_conn = multiprocessing.Pipe()

                    process = \
                        multiprocessing.Process(
                            target=solve, args=(incumbent, metric, engine,
                                                mode, pddl_problem,
                                                child_conn))
                    process.start()
                    child_conn.close()
                    pr_start = time.time()
                    process_pid = process.pid
                    uproc = psutil.Process(process_pid)

                    # Managing Processes and Memory, making sure system does
                    # not crash
                    jobid = os.getenv("SLURM_JOB_ID")
                    if jobid is not None:
                        try:
                            mem_res = subprocess.run(
                                ["scontrol", "show", "job", jobid],
                                capture_output=True, text=True, check=True
                            )
                            for line in mem_res.stdout.split("\n"):
                                if "mem=" in line:
                                    if 'G' in line:
                                        mult = 1000
                                    else:
                                        mult = 1
                                    mem = line.split(',')[1]
                                    mem = int(mem.split('=')[1][:-1]) * mult
                                    print('Allocated Memory is:', mem,
                                          flush=True)

                        except Exception as e:
                            print(f"Error retrieving allocated memory: {e}",
                                  flush=True)
                            mem = None
                    else:
                        mem = psutil.virtual_memory().total / (1024 * 1024)

                    while (time.time() - pr_start <= planner_timelimit + 1
                           and uproc.is_running()
                           and uproc.status() != psutil.STATUS_ZOMBIE) or (
                               time.time() - pr_start >= planner_timelimit):
                        time.sleep(1)
                        kill_times = [20, 30, 40]
                        mem_used = uproc.memory_info().rss / (1024 * 1024)
                        for child in uproc.children(recursive=True):
                            mem_used += uproc.memory_info().rss / (1024 * 1024)
                        if mem_used > 0.5 * mem:
                            for i in range(3):
                                try:
                                    self.trigger_gc_for_children(process_pid)
                                except Exception as e:
                                    print(e)
                                self.kill_process_tree(process_pid,
                                                       kill_times[i])
                            try:
                                process.kill()
                            except Exception as e:
                                print(e)
                                pass
                            continue

                    f = None

                    if parent_conn.poll(1):
                        try:
                            f = parent_conn.recv()
                        except Exception as e:
                            print(e)

                    process.join()

                except (AssertionError, NotImplementedError,
                        UPProblemDefinitionError) as err:
                    if self.verbose:
                        print('\n** Error in planning engine!')
                        print(err)
                    if metric == 'runtime':
                        f = planner_timelimit
                    elif metric == 'quality':
                        f = crash_cost

                if metric == 'runtime':
                    if f is None or f == 'unsolvable':
                        f = planner_timelimit
                    elif 'measure':
                        f = timeit.default_timer() - start
                        if f > planner_timelimit:
                            f = planner_timelimit
                if metric == 'quality' and (f is None or f == 'unsolvable'):
                    f = crash_cost

                if f is not None:
                    avg_f += f
                else:
                    if metric == 'quality':
                        avg_f += self.crash_cost
                    elif metric == 'runtime':
                        avg_f += planner_timelimit
                if metric == 'runtime':
                    if self.verbose:
                        print(f'\nFeedback on instance {inst}:\n\n', f, '\n')
                        print(underline)
                elif metric == 'quality':
                    if f is not None:
                        if self.verbose:
                            print(f'\nFeedback on instance {inst}:\n\n', f,
                                  '\n')
                            print(underline)
                    else:
                        if self.verbose:
                            print(f'\nFeedback on instance {inst}:\n\n', None,
                                  '\n')
                            print(underline)

            if nr_inst != 0:
                avg_f = avg_f / nr_inst
                if metric == 'runtime':
                    print(f'\nAverage performance on {nr_inst} instances:',
                          avg_f, 'seconds\n')
                if metric == 'quality':
                    print(f'''\nAverage quality performance on {nr_inst} 
                          instances:''',
                          avg_f, '\n')
                return avg_f
            else:
                print('\nPerformance could not be evaluated. No plans found.')
                return None
        else:
            return None

    def save_config(self, path, config, gaci, engine, plantype):
        """
        Save configuration in json file.

        :param path: Path where to save.
        :type path: str
        :param config: Configuration to save.
        :type config: dict
        :param gaci: AC interface object.
        :type gaci: ACInterface
        :param engine: Engine name.
        :type engine: str
        """
        if config is not None:
            config = gaci.transform_conf_from_ac(engine, config, plantype)
            with open(f'{path}/incumbent_{engine}.json', 'w') as f:
                json.dump(config, f)
            if self.verbose:
                print('\nSaved best configuration in ' +
                      f'{path}/incumbent_{engine}.json\n')
        else:
            if self.verbose:
                print(f'No configuration was saved. It was {config}')
