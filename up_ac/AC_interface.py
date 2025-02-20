"""Generic algorithm configuration interface for unified planning."""
import unified_planning
from unified_planning.environment import get_environment
from unified_planning.shortcuts import *

from tarski.io import PDDLReader as treader
import logging
import traceback
import os
import sys

from ConfigSpace.read_and_write import pcs

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

from utils.ac_feedback import qaul_feedback, runtime_feedback
from utils.patches import patch_pcs
from utils.timeout import linux_timeout
from utils.parametrizable_engines import parametrizable_engines

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

    def compute_plan_cost(self, problem, res):
        """
        Compute plan cost based on up problem and result.

        :param problem: up problem
        :type problem: up.model.Problem
        :param res: Result provided by a planning engine
        :type res: up.engines.results.Result

        :returns: Cost of the plan
        :type: int
        """
        try:    
            if not problem._metrics:
                total_cost = len(res.plan.actions)

            else:
                total_cost = 0

                if problem._metrics[0].is_minimize_action_costs():
                    action_costs = {
                        k.name: v
                        for k, v in problem._metrics[0].costs.items()
                    }

                    action_costs = {
                        k: v.constant_value()
                        for k, v in action_costs.items()
                    }

                    # Iterate actions in plan
                    for action_instance in res.plan.actions:
                        action_name = action_instance.action.name
                        total_cost += (
                            action_costs[action_name]
                            if action_name in action_costs
                            else 0
                        )
                else:
                    cost_fluent = next(
                        (fl for fl in problem.fluents
                            if "total-cost" in fl.name.lower()
                            and fl.type.is_real_type()),
                        None
                    )
                    
                    if cost_fluent is None:
                        raise ValueError("""No total-cost fluent found in the
                            problem definition.""")

                    cost_fluent_fnode = None
                    for fnode in problem.initial_values.keys():
                        if fnode.is_fluent_exp() and fnode.fluent() \
                                == cost_fluent:
                            cost_fluent_fnode = fnode
                            break

                    if cost_fluent_fnode is None:
                        raise ValueError(f"""Could not find an instance of
                            fluent {cost_fluent.name} in the initial state.""")

                    initial_cost = \
                        problem.initial_values.get(cost_fluent_fnode, 0.0)

                    initial_values = {}
                    for key, val in problem.initial_values.items():
                        initial_values[str(key)] = val

                    total_cost = initial_cost.constant_value()

                    for action_instance in res.plan.actions:
                        action = action_instance.action
                        substitutions = \
                            dict(zip(action.parameters,
                                     action_instance.actual_parameters))

                        for effect in action.effects:
                            if effect.fluent.fluent() == cost_fluent:
                                # Substitute action param to effect expression
                                effect_value_expr = \
                                    effect.value.substitute(substitutions)
                                try:
                                    evaluated_value = \
                                        problem.initial_values.get(
                                            effect_value_expr,
                                            None).constant_value()
                                    if evaluated_value is None:
                                        evaluated_value = \
                                            effect_value_expr.constant_value()
                                except AssertionError:
                                    print(f"""Warning: Could not evaluate
                                        effect value {effect_value_expr},
                                        using 0.""")
                                    evaluated_value = 0

                                if effect.is_assignment():
                                    total_cost = evaluated_value
                                elif effect.is_increase():
                                    total_cost += evaluated_value
                                elif effect.is_decrease():
                                    total_cost -= evaluated_value

            return total_cost
        except Exception:
            traceback.print_exc()
            return None

    def run_engine_config(self, config, metric, engine, plantype, problem,
                          timelimit, gb_listener=None):
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
        :param gb_listener: True if using a gray box approach (optional).
        :type gb_listener: bool

        :returns: Result from the configured engine run.
        :rtype: object

        :raises ValueError: If an unsupported planning type is provided.
        """
        if plantype == 'OneshotPlanner':
            config = self.transform_conf_from_ac(engine, config, plantype)
            if gb_listener is not None:
                with OneshotPlanner(name=engine,
                                    params=config,
                                    output_stream=gb_listener) as planner:
                    try:
                        planner = \
                            linux_timeout(planner, engine, timelimit, config)
                        if engine == 'tamer' or engine == 'pyperplan':
                            result = planner.solve(problem)
                        else:
                            result = planner.solve(problem, timeout=timelimit)
                        if (result.status ==
                                up.engines.PlanGenerationResultStatus.
                                SOLVED_SATISFICING or result.status ==
                                up.engines.PlanGenerationResultStatus.
                                SOLVED_OPTIMALLY):
                            if self.verbose:
                                print("Result found.\n")
                            if (engine == 'tamer' or engine == 'pyperplan'):
                                if metric == 'quality':
                                    feedback = self.compute_plan_cost(problem,
                                                                      result)
                                else:
                                    feedback = self.get_feedback(engine,
                                                                 metric,
                                                                 result)
                            else:
                                feedback = self.get_feedback(engine,
                                                             metric,
                                                             result)
                        elif result.status == \
                                up.engines.PlanGenerationResultStatus.\
                                UNSOLVABLE_PROVEN:
                            if self.verbose:
                                print("Problem proven unsolvable.\n")
                            feedback = 'unsolvable'
                        elif result.status == \
                                up.engines.PlanGenerationResultStatus.\
                                INTERNAL_ERROR:
                            if self.verbose:
                                print("Internal Error.\n")
                            feedback = None
                        else:
                            if self.verbose:
                                print("No plan found.\n")
                            feedback = None
                    except Exception as e:
                        if self.verbose:
                            print("No plan found.\n")
                            print('Exception:', e)
                        feedback = None
            else:
                
                with OneshotPlanner(name=engine,
                                    params=config) as planner:
                    try:
                        planner = \
                            linux_timeout(planner, engine, timelimit, config)
                        if engine == 'tamer' or engine == 'pyperplan':
                            result = planner.solve(problem)
                        else:
                            result = planner.solve(problem, timeout=timelimit)
                        if (result.status ==
                                up.engines.PlanGenerationResultStatus.
                                SOLVED_SATISFICING or result.status ==
                                up.engines.PlanGenerationResultStatus.
                                SOLVED_OPTIMALLY):
                            if self.verbose:
                                print("Result found.\n")
                            if (engine == 'tamer' or engine == 'pyperplan'):
                                if metric == 'quality':
                                    feedback = self.compute_plan_cost(problem,
                                                                      result)
                                else:
                                    feedback = self.get_feedback(engine,
                                                                 metric,
                                                                 result)
                            else:
                                feedback = self.get_feedback(engine,
                                                             metric,
                                                             result)
                        elif result.status == \
                                up.engines.PlanGenerationResultStatus.\
                                UNSOLVABLE_PROVEN:
                            if self.verbose:
                                print("Problem proven unsolvable.\n")
                            feedback = 'unsolvable'
                        elif result.status == \
                                up.engines.PlanGenerationResultStatus.\
                                INTERNAL_ERROR:
                            if self.verbose:
                                print("Internal Error.\n")
                            feedback = None
                        else:
                            if self.verbose:
                                print("No plan found.\n")
                            feedback = None
                    except Exception as e:
                        if self.verbose:
                            print("No plan found.\n")
                            print('Exception:', e)
                        feedback = None

        elif plantype == 'AnytimePlanner':
            config = self.transform_conf_from_ac(engine, config, plantype)
            parametrizable_engines(engine)
            if gb_listener is not None:
                with AnytimePlanner(name=engine,
                                    params=config,
                                    output_stream=gb_listener) as planner:
                    try:
                        planner = linux_timeout(planner, engine,
                                                timelimit, config)
                        results = planner.get_solutions(problem,
                                                        timeout=timelimit)
                        for res in results:
                            if res is not None and \
                                (res.status ==
                                    up.engines.PlanGenerationResultStatus.
                                    SOLVED_SATISFICING or res.status ==
                                    up.engines.PlanGenerationResultStatus.
                                    SOLVED_OPTIMALLY):
                                result = res
                                if self.verbose:
                                    print("Result found.\n")
                                feedback = self.compute_plan_cost(problem, res)
                                results.close()
                                break
                            elif res is not None and \
                                    res.status == \
                                    up.engines.PlanGenerationResultStatus.\
                                    UNSOLVABLE_PROVEN:
                                if self.verbose:
                                    print("Problem proven unsolvable.\n")
                                results.close()
                                feedback = 'unsolvable'
                                break
                            elif res is not None and \
                                res.status == \
                                    up.engines.PlanGenerationResultStatus.\
                                    INTERNAL_ERROR:
                                if self.verbose:
                                    print("Internal Error.\n")
                                results.close()
                                feedback = None
                                break
                            elif res is not None and \
                                res.status == \
                                    up.engines.PlanGenerationResultStatus.\
                                    INTERMEDIATE:
                                feedback = self.compute_plan_cost(problem, res)
                                results.close()
                                break
                            else:
                                if self.verbose:
                                    print("No plan found.\n")
                                results.close()
                                feedback = None
                                break
                    except Exception as e:
                        if self.verbose:
                            print("No plan found.\n")
                            print('Exception:', e)
                        try:
                            results.close()
                        except Exception:
                            pass
                        feedback = None
            else:
                with AnytimePlanner(name=engine,
                                    params=config) as planner:
                    try:
                        planner = linux_timeout(planner, engine,
                                                timelimit, config)
                        results = planner.get_solutions(problem,
                                                        timeout=timelimit)
                        for res in results:
                            if res is not None and \
                                (res.status ==
                                    up.engines.PlanGenerationResultStatus.
                                    SOLVED_SATISFICING or res.status ==
                                    up.engines.PlanGenerationResultStatus.
                                    SOLVED_OPTIMALLY):
                                if self.verbose:
                                    print("Result found.\n")
                                feedback = self.compute_plan_cost(problem, res)
                                results.close()
                                break
                            elif res is not None and \
                                    res.status == \
                                    up.engines.PlanGenerationResultStatus.\
                                    UNSOLVABLE_PROVEN:
                                if self.verbose:
                                    print("Problem proven unsolvable.\n")
                                feedback = 'unsolvable'
                                results.close()
                                break
                            elif res is not None and \
                                res.status == \
                                    up.engines.PlanGenerationResultStatus.\
                                    INTERNAL_ERROR:
                                if self.verbose:
                                    print("Internal Error.\n")
                                feedback = None
                                results.close()
                                break
                            elif res is not None and \
                                res.status == \
                                    up.engines.PlanGenerationResultStatus.\
                                    INTERMEDIATE:
                                feedback = self.compute_plan_cost(problem, res)
                                results.close()
                                break
                            else:
                                if self.verbose:
                                    print("No plan found.\n")
                                feedback = None
                                results.close()
                                break
                    except Exception as e:
                        if self.verbose:
                            print("No plan found.\n")
                            print('Exception:', e)
                        feedback = None
                        try:
                            results.close()
                        except Exception:
                            pass

        return feedback
