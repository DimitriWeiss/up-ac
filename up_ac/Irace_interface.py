"""Irace algorithm configuration interface for unified planning."""
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

from up_ac.AC_interface import GenericACInterface
from up_ac.utils.pcs_transform import transform_pcs

from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.conditions import (
    InCondition,
    AndConjunction,
    EqualsCondition,
)


class IraceInterface(GenericACInterface):
    """Irace AC interface."""

    def __init__(self):
        """Initialize Irace interface."""
        GenericACInterface.__init__(self)

    def transform_conf_from_ac(self, engine, configuration):
        """
        Transform configuration from algorithm configuration format to engine format.

        This function takes a configuration in the algorithm configuration format, specific to the provided engine,
        and transforms it into the corresponding format for the given engine.

        :param str engine: Name of the engine for which the configuration is being transformed.
        :param dict configuration: A dictionary containing parameter names with their values.

        :return: A dictionary containing the transformed configuration for the specified engine.
        :rtype: dict

        :raises ValueError: If the specified engine is not supported.

        :note:
            The transformation process varies based on the engine type and specific configurations.
        """
        config = transform_pcs(engine, configuration)
        if engine == 'lpg':
            del_list = []
            add_list = []
            for pname, pvalue in config.items():
                if pname in self.engine_param_types['lpg']:
                    if self.engine_param_types['lpg'][pname] == 'FLAGS':
                        del_list.append(pname)
                        flag_pname = pname + '=' + config[pname]
                        add_list.append(flag_pname)
                    elif self.engine_param_types['lpg'][pname] == 'FLAG':
                        if config[pname] == '1':
                            config[pname] = ''
                        else:
                            del_list.append(pname)

            for dl in del_list:
                del config[dl]

            for al in add_list:
                config[al] = ''

        if engine == 'fast-downward' or engine == 'symk':

            evals = ['eager_greedy', 'eager_wastar',
                     'lazy_greedy', 'lazy_wastar']
            open_eval = ['epsilon_greedy', 'single']
            open_evals = ['pareto', 'tiebreaking',
                          'type_based']
            pruning = ['atom_centric_stubborn_sets']

            if len(config) == 1:
                pass
            else:
                search_option = config['fast_downward_search_config'] + '('                    
                if 'evaluator' in config:
                    if config['evaluator'] in evals:
                        search_option += \
                            '[' + str(config['evaluator']) + '()], '
                    else:
                        search_option += str(config['evaluator']) + '(), '

                if 'open' in config:
                    if config['open'] not in open_eval and \
                            config['open'] not in open_evals:
                        search_option += '[' + str(config['open']) + '()], '
                    elif config['open'] in open_eval:
                        search_option += \
                            '[' + str(config['open']) + '(' + \
                            str(config['open_list_evals']) + ')], '
                    elif config['open'] in open_evals:
                        search_option += '[' + str(config['open']) + '([]' + \
                            str(config['open_list_evals']) + '])], '

                if 'evaluator' in config:
                    if config['evaluator'] == 'ehc':
                        search_option += 'preferred_usage=' + \
                            str(config['ehc_preferred_usage']) + ','

                if 'reopen_closed' in config:
                    search_option += 'reopen_closed=' + \
                        str(config['reopen_closed']) + ','

                if 'randomize_successors' in config:
                    search_option += 'randomize_successors=' + \
                        str(config['randomize_successors']) + ','

                if 'pruning' in config:
                    if config['pruning'] in pruning:
                        search_option += 'pruning=' + \
                            str(config['pruning'])  # + '(use_sibling_shortcut=' \
                        '''
                            + config[
                                'atom_centric_stubborn_sets_use_sibling'] + \
                            ',atom_selection_strategy=' + \
                            config['atom_selection_strategy'] + '(), '
                        '''
                    else:
                        search_option += \
                            'pruning=' + str(config['pruning']) + '(),'

                if 'cost_type' in config:
                    search_option += 'cost_type=' + config['cost_type'] + ')'
                else:
                    search_option += ')'
                search_option = search_option.replace(" ", "")

                if engine == 'fast-downward':
                    config = {'fast_downward_search_config': search_option}
                else:
                    config = {'symk_search_config': search_option}

        else:
            config = configuration

        return config

    def get_ps_irace(self, param_space):
        """
        Retrieve parameter space information for configuring irace.

        :param param_space: The ConfigSpace object defining the parameter space.
        :type param_space: ConfigSpace.ConfigurationSpace

        :return: A tuple containing:
            - dict: Default values for parameters.
            - bool: Indicates if there are forbidden parameter value combinations.
        :rtype: tuple
        """


        def set_conditional(c, parent, cond_params):
            """
            Set conditions as strings for irace parameter space.

            :param c: The condition to be set.
            :type c: ConfigSpace.conditions
            :param parent: The parent parameter for the condition.
            :type parent: ConfigSpace.Parameter
            :param cond_params: A dictionary to store conditions as strings.
            :type cond_params: dict

            :return: A dictionary with conditions as strings.
            :rtype: dict
            """
            if isinstance(c, InCondition):
                if isinstance(parent, CategoricalHyperparameter):
                    cond_params[c.child.name] = \
                        f' | {parent.name} %in% c({str(c.values)[1:-1]})'
                elif isinstance(parent, UniformFloatHyperparameter):
                    cond_params[c.child.name] = \
                        f' | {parent.name} %in% r({str(c.values)[1:-1]})'
                elif isinstance(parent, UniformIntegerHyperparameter):
                    cond_params[c.child.name] = \
                        f' | {parent.name} %in% i({str(c.values)[1:-1]})'
            elif isinstance(c, EqualsCondition):
                if isinstance(parent, CategoricalHyperparameter):
                    cond_params[c.child.name] = \
                        f' | {parent.name} == "{c.value}"'
                elif isinstance(parent, UniformFloatHyperparameter):
                    cond_params[c.child.name] = \
                        f' | {parent.name} == {c.value}'
                elif isinstance(parent, UniformIntegerHyperparameter):
                    cond_params[c.child.name] = \
                        f' | {parent.name} == {c.value})'
            return cond_params

        params = param_space.get_hyperparameters_dict()
        conditions = param_space.get_conditions()
        cond_params = {}
        for c in conditions:
            parent = c.get_parents()
            parent = parent[0]
            if isinstance(c, AndConjunction):
                cond_params = set_conditional(c, parent, cond_params)
                for cc in c.components[1:]:
                    parent = cc.get_parents()
                    parent = parent[0]
                    and_cond = set_conditional(cc, parent, cond_params)
                    cond_params[cc.child.name] += \
                        ' &&' + and_cond[cc.child.name][2:]
            else:
                cond_params = set_conditional(c, parent, cond_params)

        names = []
        values = []
            
        for _, param in params.items():
            names.append(param.name)
            values.append(param.default_value)

        default_conf = pd.DataFrame([values], columns=names)

        with (ro.default_converter + pandas2ri.converter).context():
            default_conf = ro.conversion.get_conversion().py2rpy(default_conf)

        irace_param_space = ''

        for p, param in params.items():
            if param.name in cond_params:
                condition = cond_params[param.name]
            else:
                condition = ''
            if isinstance(param, CategoricalHyperparameter):
                choices = ''
                for pc in param.choices:
                    choices += f'\"{pc}\", '
                irace_param_space += '\n' + param.name + \
                    ' \"\" c ' + f'({choices[:-2]})' + condition
            elif isinstance(param, UniformFloatHyperparameter):
                irace_param_space += '\n' + param.name + \
                    ' \"\" r ' + f'({param.lower}, {param.upper})' + condition
            elif isinstance(param, UniformIntegerHyperparameter):
                irace_param_space += '\n' + param.name + \
                    ' \"\" i ' + f'({param.lower}, {param.upper})' + condition

        forbidden = ''
        for f in param_space.forbidden_clauses:
            fpair = f.get_descendant_literal_clauses()
            if isinstance(
                fpair[0].hyperparameter, CategoricalHyperparameter) and\
                    isinstance(
                        fpair[1].hyperparameter, CategoricalHyperparameter):
                forbidden += '\n(' + fpair[0].hyperparameter.name + ' == ' + \
                    f'\"{fpair[0].value}\") ' +\
                    '& (' + fpair[1].hyperparameter.name + ' == ' + \
                    f'\"{fpair[1].value}\")'
            elif isinstance(
                    fpair[0].hyperparameter, CategoricalHyperparameter):
                forbidden += '\n(' + fpair[0].hyperparameter.name + ' == ' + \
                    f'\"{fpair[0].value}\") ' +\
                    '& (' + fpair[1].hyperparameter.name + ' == ' + \
                    f'{fpair[1].value})'
            elif isinstance(
                    fpair[1].hyperparameter, CategoricalHyperparameter):
                forbidden += '\n(' + fpair[0].hyperparameter.name + ' == ' + \
                    f'{fpair[0].value}) ' +\
                    '& (' + fpair[1].hyperparameter.name + ' == ' + \
                    f'\"{fpair[1].value}\")'

        forbidden += '\n'

        if len(forbidden) > 2:
            forbiddens = True
        else:
            forbiddens = False

        with open("forbidden.txt", "w") as text_file:
            text_file.write(forbidden)

        self.irace_param_space = irace_param_space

        return default_conf, forbiddens
