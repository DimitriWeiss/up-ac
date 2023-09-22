"""Functionalities for managing and calling configurators."""
from up_ac.AC_interface import *
from up_ac.configurators import Configurator

import timeit
import os
from threading import Thread
from queue import Queue
import subprocess
import dill 
import shutil
from unified_planning.exceptions import UPProblemDefinitionError
from pebble import concurrent
from concurrent.futures import TimeoutError


class OATConfigurator(Configurator):
    """Configurator functions."""

    def __init__(self):
        """Initialize OAT configurator."""
        Configurator.__init__(self)

    def get_OAT_incumbent(self):

        path = self.scenario['path_to_OAT']
        read_param = False
        config = {}
        with open(f'{path}tunerLog.txt', 'r') as f:
            for line in f:
                line = line.split(' ')
                for i, l in enumerate(line):
                    line[i] = l.replace(' ', '')
                if 'results' in line:
                    read_param = False
                if read_param:
                    param_name = str(line[0].replace('\t', '')[:-1])
                    config[param_name] = line[1].replace('\n', '')      
                if 'according' in line:
                    read_param = True
                    
        return config

    def get_feedback_function(self, gaci, engine, metric, mode,
                              gray_box=False):
        """
        Generate the function to run engine and get feedback.

        parameter gaci: AC interface object.
        parameter engine: str, engine name.
        parameter metric: str, 'runtime' or 'quality'
        parameter mode: str, type of planning.
        parameter gray_box: True, if gra box to use

        return planner_feedback: function, planner feedback function.
        """
        if engine in self.capabilities[metric][mode]:
            self.metric = metric

            if gray_box:
                class gb_out():
                    def __init__(self, q, res):
                        self.q = q
                        self.res = res

                    def write(self, txt):
                        # TODO
                        # pass output to configurator
                        if self.res.empty():
                            self.q.put(txt)
                q = Queue()
                res = Queue()
                gb_out = gb_out(q, res)

            def planner_feedback(config, instance, reader):

                self.reader = reader 
                
                start = timeit.default_timer()
                instance_p = f'{instance}'
                domain_path = instance_p.rsplit('/', 1)[0]
                domain = f'{domain_path}/domain.pddl'
                pddl_problem = self.reader.parse_problem(f'{domain}',
                                                         f'{instance_p}')
                # gray box in OAT only works with runtime scenarios
                if gray_box:
                    def planner_thread(gb_out, problem, res, ac_tool,
                                       config, metric, engine, mode, 
                                       pddl_problem):
                        res.put(
                            gaci.run_engine_config(config,
                                                   metric,
                                                   engine,
                                                   mode,
                                                   pddl_problem,
                                                   gb_out))

                    thread = Thread(target=planner_thread,
                                    args=(gb_out, problem, res, ac_tool,
                                          config, metric, engine, mode, 
                                          pddl_problem),
                                    daemon=True)

                    thread.start()

                    while thread.is_alive():
                        try:
                            output = q.get(False)
                        except:
                            output = None
                        if output is not None and len(output) not in (0, 1):
                            print('gray box:', output)
                        if not res.empty():
                            thread.join()

                    feedback = res.get()

                else:
                    feedback = \
                        gaci.run_engine_config(config,
                                               metric,
                                               engine,
                                               mode,
                                               pddl_problem)
                                               
                    try:
                        @concurrent.process(timeout=self.scenario['timelimit'])
                        def solve(config, metric, engine,
                                  mode, pddl_problem):
                            feedback = \
                                gaci.run_engine_config(config,
                                                       metric, engine,
                                                       mode, pddl_problem)

                            return feedback

                        feedback = solve(config, metric, engine,
                                         mode, pddl_problem)
                    
                        try:
                            feedback = feedback.result()
                        except TimeoutError:
                            if metric == 'runtime':
                                feedback = self.planner_timelimit
                            elif metric == 'quality':
                                feedback = self.crash_cost

                    except (AssertionError, NotImplementedError,
                            UPProblemDefinitionError):
                        print('\n** Error in planning engine!')
                        if metric == 'runtime':
                            feedback = self.planner_timelimit
                        elif metric == 'quality':
                            feedback = self.crash_cost

                if feedback is not None:
                    if metric == 'quality':
                        self.print_feedback(engine, instance_p, feedback)
                        return -feedback
                    elif metric == 'runtime':
                        if engine in ('tamer', 'pyperplan'):
                            feedback = timeit.default_timer() - start
                            self.print_feedback(engine, instance_p, feedback)
                        else:
                            feedback = feedback
                            self.print_feedback(engine, instance_p, feedback)
                        return feedback
                else:
                    # Penalizing failed runs
                    if metric == 'runtime':
                        # Penalty is max runtime in runtime scenario
                        feedback = self.scenario['timelimit']
                        self.print_feedback(engine, instance_p, feedback)
                    else:
                        # Penalty is defined by user in quality scenario
                        feedback = self.crash_cost
                        self.print_feedback(engine, instance_p, feedback)

                    return feedback

            path_to_OAT = 'path_to_OAT'
            dill.dump(
                planner_feedback, open(
                    f'{self.scenario[path_to_OAT]}feedback.pkl', 'wb'),
                recurse=True)

            planner_feedback = f'{self.scenario[path_to_OAT]}call_engine_OAT.py'

            return planner_feedback
        else:
            print(f'Algorithm Configuration for {metric} of {engine} in' + \
                  ' {mode} is not supported.')
            return None

    def set_scenario(self, engine, param_space, gaci,
                     configuration_time=120, n_trials=400, min_budget=1,
                     max_budget=3, crash_cost=0, planner_timelimit=30,
                     n_workers=1, instances=[], instance_features=None,
                     metric='runtime', popSize=128, evlaLimit=2147483647):
        """
        Set up algorithm configuration scenario.

        parameter ac_tool: str, which configuration tol.
        parameter engine: str, which engine.
        parameter param_space: ConfigSpace object.
        parameter gaci: AC interface object.
        parameter configuration_time: int, overall configuration time budget.
        parameter n_trials: int, max number of engine evaluations.
        parameter min_budget: int, min number of instances to use.
        parameter max_budget: int, max number of instances to use.
        parameter crash_cost: int, which cost to use if engine fails.
        parameter planner_timelimit: int, max runtime per evaluation.
        parameter n_workers: int, no. of cores to utilize.
        parameter instances: list, problem instance paths.
        parameter instance_features: dict, inst names and lists of features.
        parameter metric: str, optimization metric.
        parameter popSize: int, population size of configs per generation (OAT).
        parameter evlaLimit: int, max no. of evaluations (OAT).
        """
        if not instances:
            instances = self.train_set
        self.crash_cost = crash_cost

        param_file = gaci.get_ps_oat(param_space)

        path = os.getcwd().rsplit('up-ac', 2)[0]
        path += 'up-ac/up_ac'

        path_to_xml = f'{path}/OAT/{engine}parameterTree.xml'

        oat_dir = f'{path}/OAT/'

        with open(path_to_xml, 'w') as xml:
            xml.write(param_file)

        inst_dir = f'{path}/OAT/{engine}'

        if os.path.isdir(inst_dir):
            shutil.rmtree(inst_dir, ignore_errors=True)

        os.mkdir(inst_dir)
        file_name = 0
        for inst in instances:
            with open(f'{inst_dir}/{file_name}.txt', 'w') as f:
                f.write(f'{inst}')
            file_name += 1

        scenario = dict(
            xml=path_to_xml,
            timelimit=planner_timelimit,
            wallclock=configuration_time,
            start_gen=min_budget,
            end_gen=max_budget,
            n_workers=n_workers,
            metric=metric,
            instance_dir=inst_dir,
            path_to_OAT=oat_dir,
            popSize=popSize,
            evlaLimit=evlaLimit
        )

        self.scenario = scenario

    def optimize(self, ac_tool, feedback_function=None, gray_box=False):
        """
        Run the algorithm configuration.

        parameter ac_tool: str, which AC tool.
        parameter feedback_function: function to run engine and get feedback.
        parameter gray_box: True, if gray box usage.
        """
        if feedback_function is not None:

            print('\nStarting Parameter optimization\n')

            if self.scenario['metric'] == 'quality':
                tunefor = ' --byValue '
            elif self.scenario['metric'] == 'runtime':
                tunefor = ' --enableRacing=true '
            
            n_workers = self.scenario['n_workers']
            path_to_OAT = self.scenario['path_to_OAT']
            param_space = self.scenario['xml']
            instance_folder = self.scenario['instance_dir']
            planner_timelimit = self.scenario['timelimit']
            min_budget = self.scenario['start_gen']
            max_budget = self.scenario['end_gen']
            evalLimit = self.scenario['evlaLimit']
            popSize = self.scenario['popSize']

            p = subprocess.Popen(['./Optano.Algorithm.Tuner.Application' +
                                  ' --master' +
                                  f' --maxParallelEvaluations={n_workers} ' + 
                                  '--basicCommand=\"python3 ' + 
                                  f'{feedback_function} {{instance}} ' + 
                                  '{{arguments}}\"' + 
                                  f' --parameterTree={param_space} ' + 
                                  '--trainingInstanceFolder' +
                                  f'=\"{instance_folder}\" ' + 
                                  f'--cpuTimeout={planner_timelimit}' + 
                                  f'{tunefor}' + 
                                  f'--numGens={max_budget} ' + 
                                  f'--goalGen={min_budget} ' +
                                  f'--instanceNumbers={min_budget}:' +
                                  f'{max_budget} ' +
                                  f'--evaluationLimit={evalLimit} ' +
                                  f'--popSize={popSize}'],
                                 stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                                 cwd=f'{path_to_OAT[:-1]}', shell=True)

            while p.poll() is None:
                line = p.stdout.readline()
                print(line.decode('utf-8'))

            self.incumbent = self.get_OAT_incumbent()

            print(f'\nBest Configuration found by {ac_tool} is:\n',
                  self.incumbent)

            return self.incumbent, None
        else:
            return None, None
