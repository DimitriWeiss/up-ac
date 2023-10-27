"""Generic algorithm configuration interface for unified planning."""
import unified_planning
from unified_planning.environment import get_environment
from unified_planning.shortcuts import *
from up_ac.utils.ac_feedback import qaul_feedback, runtime_feedback
from up_ac.utils.patches import patch_pcs
from tarski.io import PDDLReader as treader
import logging

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
        self.verbose = True

    def get_available_engines(self):
        """Get planning engines installed in up."""
        factory = unified_planning.engines.factory.Factory(self.environment)

        return factory.engines

    def compute_instance_features(self, domain, instance):
        """
        Compute instance features of a given PDDL instance.

        :param domain: PDDL string representing the problem domain.
        :type domain: str
        :param instance: PDDL string representing the problem instance.
        :type instance: str

        :returns: Computed instance features.
        :rtype: list
        """
        logging.getLogger().setLevel(logging.ERROR)
        try:
            # TODO catch duplicate errors in tarski
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
        """
        Read parameter configuration space (PCS) files for specified engines.

        :param engines: Names of the engines.
        :type engines: list of str
        :param pcs_dir: Path to the directory containing the PCS files.
        :type pcs_dir: str
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
        """
        Get feedback from a planning engine after a run.

        :param engine: Name of the planning engine.
        :type engine: str
        :param fbtype: Type of feedback: 'quality' or 'runtime'.
        :type fbtype: str
        :param result: Planning result.
        :type result: object

        :returns: Feedback based on the specified feedback type.
        :rtype: object

        :raises ValueError: If an unsupported feedback type is provided.
        """
        if fbtype == 'quality':
            feedback = qaul_feedback(engine, result)
        if fbtype == 'runtime':
            feedback = runtime_feedback(engine, result)

        return feedback

    def run_engine_config(self, config, metric, engine,
                          plantype, problem, gray_box_listener=None):
        """
        Execute a configured engine run.

        :param config: Configuration of the engine.
        :type config: dict
        :param metric: Metric for the evaluation: 'runtime' or 'quality'.
        :type metric: str
        :param engine: Name of the engine.
        :type engine: str
        :param plantype: Type of planning: 'OneshotPlanner' or 'AnytimePlanner'.
        :type plantype: str
        :param problem: Path to the problem instance.
        :type problem: str
        :param gray_box_listener: True if using a gray box approach (optional).
        :type gray_box_listener: bool

        :returns: Result from the configured engine run.
        :rtype: object

        :raises ValueError: If an unsupported planning type is provided.
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
                            if self.verbose:
                                print("Result found.\n")
                        else:
                            if self.verbose:
                                print("No plan found.\n")
                        feedback = self.get_feedback(engine, metric, result)
                    except:
                        if self.verbose:
                            print("No plan found.\n")
                        feedback = None
            else:
                with OneshotPlanner(name=engine,
                                    params=config) as planner:
                    if self.verbose:
                        print(config)
                    # result = planner.solve(problem)
                    
                    try:
                        result = planner.solve(problem)
                        if (result.status ==
                                up.engines.PlanGenerationResultStatus.
                                SOLVED_SATISFICING):
                            if self.verbose:
                                print("Result found.\n")
                        else:
                            if self.verbose:
                                print("No plan found.\n")
                        feedback = self.get_feedback(engine, metric, result)
                    except:
                        if self.verbose:
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
                            if self.verbose:
                                print("Result found.\n")
                        else:
                            if self.verbose:
                                print("No plan found.\n")
                        feedback = self.get_feedback(engine, metric, result)
                    except:
                        if self.verbose:
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
                            if self.verbose:
                                print("Result found.\n")
                        else:
                            if self.verbose:
                                print("No plan found.\n")
                        feedback = self.get_feedback(engine, metric, result)
                    except:
                        if self.verbose:
                            print("No plan found.\n")
                        feedback = None

        return feedback
