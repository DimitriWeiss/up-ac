"""Generic algorithm configuration interface for unified planning."""
import unified_planning
from unified_planning.environment import get_environment
from unified_planning.shortcuts import *
from up_ac.utils.ac_feedback import qaul_feedback, runtime_feedback
from up_ac.utils.patches import patch_pcs
from tarski.io import PDDLReader as treader

from ConfigSpace.read_and_write import pcs

pcs = patch_pcs(pcs)


class GenericACInterface():
    """Generic AC interface."""

    def __init__(self):
        """Initialize generic interface."""
        self.environment = get_environment()
        self.available_engines = self.get_available_engines()
        self.engine_param_spaces = {}
        self.engine_param_types = {}
        self.treader = treader(raise_on_error=True)

    def get_available_engines(self):
        """Get planning engines installed in up."""
        factory = unified_planning.engines.factory.Factory(self.environment)

        return factory.engines

    def compute_instance_features(self, domain, instance):
        """Compute some instance features of a iven pddl instance.

        parameter domain: pddl, problem domain.
        parameter instance: pddl, problem instance.

        return features: list, computed instance features
        """
        try:
            # TODO catch duplicte errors in tarski
            features = []
            self.treader.parse_domain(domain)
            problem = self.treader.parse_instance(instance)
            lang = problem.language
            features.append(len(lang.predicates))
            features.append(len(lang.functions))
            features.append(len(lang.constants()))
            features.append(len(list(problem.actions)))
            features.append(features[1] / features[0])
            features.append(features[1] / features[2])
            features.append(features[1] / features[3])
            features.append(features[0] / features[2])
            features.append(features[0] / features[3])
            features.append(features[2] / features[3])
        except:
            features = [0 for _ in range(10)]

        return features

    def read_engine_pcs(self, engines, pcs_dir):
        """Read in pcs file for engine.

        parameter engines: list of str, names of engines.
        parameter pcs_dir: str, path to directory with pcs files.
        """
        if pcs_dir[-1] != '/':
            pcs_dir = pcs_dir + '/'

        for engine in engines:
            with open(pcs_dir + engine + '.pcs', 'r') as f:
                self.engine_param_spaces[engine] = pcs.read(f)

            with open(pcs_dir + engine + '.pcs', 'r') as f:
                lines = f.readlines()
                self.engine_param_types[engine] = {}
                for line in lines:
                    if '# FLAGS #' in line:
                        self.engine_param_types[engine][
                            '-' + line.split(' ')[0]] = 'FLAGS'
                    elif '# FLAG' in line:
                        self.engine_param_types[engine][
                            '-' + line.split(' ')[0]] = 'FLAG'

    def get_feedback(self, engine, fbtype, result):
        """Get feedback from planning engine after run.

        parameter engine: str, name of planning engine.
        parameter fbtype: str, type of feedback
        parameter result: object, planning result.
        """
        if fbtype == 'quality':
            feedback = qaul_feedback(engine, result)
        if fbtype == 'runtime':
            feedback = runtime_feedback(engine, result)

        return feedback

    def run_engine_config(self, config, metric, engine,
                          plantype, problem, gray_box_listener=None):
        """Execute configurated engine run.

        parameter config: configuration of engine.
        parameter metric: str, 'runtime' or 'quality'
        parameter engine: str, engine name.
        parameter plantype: str, type of planning.
        parameter problem: str, path to problem instance
        parameter gray_box_listener: True, if gra box to use

        return feedback: result from configurated engine run.
        """
        if plantype == 'OneshotPlanner':
            config = self.transform_conf_from_ac(engine, config)
            if gray_box_listener is not None:
                with OneshotPlanner(name=engine,
                                    params=config,
                                    output_stream=gray_box_listener) as planner:
                    try:
                        result = planner.solve(problem)
                        if (result.status ==
                                up.engines.PlanGenerationResultStatus.
                                SOLVED_SATISFICING):
                            print("Result found.\n")
                        else:
                            print("No plan found.\n")
                        feedback = self.get_feedback(engine, metric, result)
                    except:
                        print("No plan found.\n")
                        feedback = None
            else:
                with OneshotPlanner(name=engine,
                                    params=config) as planner:
                    result = planner.solve(problem)
                    try:
                        result = planner.solve(problem)
                        if (result.status ==
                                up.engines.PlanGenerationResultStatus.
                                SOLVED_SATISFICING):
                            print("Result found.\n")
                        else:
                            print("No plan found.\n")
                        feedback = self.get_feedback(engine, metric, result)
                    except:
                        print("No plan found.\n")
                        feedback = None

        elif plantype == 'AnytimePlanner':
            config = self.transform_conf_from_ac(engine, config)
            if gray_box_listener is not None:
                with AnytimePlanner(name=engine,
                                    params=config,
                                    output_stream=gray_box_listener) as planner:
                    try:
                        result = planner.solve(problem)
                        if (result.status ==
                                up.engines.PlanGenerationResultStatus.
                                SOLVED_SATISFICING):
                            print("Result found.\n")
                        else:
                            print("No plan found.\n")
                        feedback = self.get_feedback(engine, metric, result)
                    except:
                        print("No plan found.\n")
                        feedback = None
            else:
                with AnytimePlanner(name=engine,
                                    params=config) as planner:
                    try:
                        result = planner.solve(problem)
                        if (result.status ==
                                up.engines.PlanGenerationResultStatus.
                                SOLVED_SATISFICING):
                            print("Result found.\n")
                        else:
                            print("No plan found.\n")
                        feedback = self.get_feedback(engine, metric, result)
                    except:
                        print("No plan found.\n")
                        feedback = None

        return feedback
