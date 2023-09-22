"""OAT algorithm configuration interface for unified planning."""
from up_ac.AC_interface import GenericACInterface
from up_ac.utils.pcs_transform import transform_pcs

from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.conditions import (
    AndConjunction
)


class OATInterface(GenericACInterface):
    """OAT AC interface."""

    def __init__(self):
        """Initialize OAT interface."""
        GenericACInterface.__init__(self)

    def transform_conf_from_ac(self, engine, configuration):
        """Transform configuration to up engine format.

        parameter engines: list of str, names of engines.
        parameter configuration: dict, parameter names with values.

        return config: dict, configuration.
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

        elif engine == 'fast-downward' or engine == 'symk':

            evals = ['eager_greedy', 'eager_wastar',
                     'lazy_greedy', 'lazy_wastar']
            open_eval = ['epsilon_greedy', 'single']
            open_evals = ['pareto', 'tiebreaking',
                          'type_based']
            pruning = ['atom_centric_stubborn_sets']

            if len(config) in (0, 1):
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
                        search_option += '[' + str(config['open']) + '(' + \
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
                            str(config['pruning']) + '(use_sibling_shortcut=' \
                            + config[
                                'atom_centric_stubborn_sets_use_sibling'] + \
                            ',atom_selection_strategy=' + \
                            config['atom_selection_strategy'] + '(), '
                    else:
                        search_option += 'pruning=' + \
                            str(config['pruning']) + '(),'

                if 'cost_type' in config:
                    search_option += 'cost_type=' + config['cost_type'] + ')'
                else:
                    search_option += ')'
                search_option = search_option.replace(" ", "")

                if engine == 'fast-downward':
                    config = {'fast_downward_search_config': search_option}
                else:
                    config = {'symk_search_config': search_option}

        elif engine in ('enhsp', 'tamer', 'pyperplan'):
            config = {}
            for param in \
                    self.engine_param_spaces[engine].get_hyperparameters():
                if isinstance(param, UniformFloatHyperparameter):
                    config[param.name] = float(configuration[param.name])
                elif isinstance(param, UniformIntegerHyperparameter):
                    config[param.name] = int(configuration[param.name])
 
        return config

    def get_ps_oat(self, param_space):
        '''
        OAT does not handle forbidden parameter value combinations.
        OAT can handle multiple parent and 1 child conditionals,
        but not one parent multiple children conditionals.
        We naively just take the first one in the list.
        OAT does not support conditionals that are conditional.
        We leave them out naively.
        Oat does not support conditionals with value ranges.
        We naively only use the first value.

        Note: Although this is suboptimal, invalid configurations will
        lead to crash or bad results such that OAT will rate them
        as subpar.

        parameter param_space: ConfigSpace object.
        '''

        param_file = '<?xml version="1.0" encoding="utf-8" ?>\n'
        param_file += \
            '<node xsi:type="and" xsi:noNamespaceSchemaLocation="' + \
            '../parameterTree.xsd" xmlns:xsi="http://www.w3.org/' + \
            '2001/XMLSchema-instance">\n'

        hyperparameters = param_space.get_hyperparameters()
        to_set = []
        for hp in hyperparameters:
            to_set.append(hp.name)

        conditions = param_space.get_conditions()

        parents = {}
        for cond in conditions:
            if not isinstance(cond, AndConjunction):
                if cond.parent.name not in parents:
                    parents[cond.parent.name] = \
                        {cond.child.name: [cond.value, cond.child]}
                else:
                    parents[cond.parent.name][cond.child.name] = \
                        [cond.value, cond.child]
            else:
                for c in cond.components:
                    if c.parent.name not in parents:
                        parents[c.parent.name] = \
                            {c.child.name: [c.value, c.child]}
                    else:
                        parents[c.parent.name][c.child.name] = \
                            [c.value, c.child]

        def set_conditionals(children, param_file, to_set, parents, tab=''):
            """
            Set conditional relations between parameters.

            parameter children: dict, child parameters.
            parameter param_file: str, OAT parameter tree to be saved in xml.
            parameter to_set: list, parameter names to still be included.
            parameter parents: dict, parent parameters.
            parameter tab: str, indicates depth of tree (\t).

            return param_file: str, OAT parameter tree to be saved in xml.
            """
            for child, value in children.items():
                if child in to_set:
                    if isinstance(value[0], list):
                        value[0] = value[0][0]
                    param_file += f'{tab}\t\t<choice>\n'
                    param_file += f'{tab}\t\t\t<string>{value[0]}</string>\n'
                    param_file += \
                        f'{tab}\t\t\t<child xsi:type="value" id="{child}">\n'
                    if isinstance(value[1], CategoricalHyperparameter):
                        choices = ''
                        for c in value[1].choices:
                            choices += f'{c} '
                        param_file += f'{tab}\t\t\t\t<domain xsi:type=' + \
                            '"categorical" strings="{choices[:-1]}" ' + \
                            'defaultIndexOrValue="{value[1].choices.' + \
                            'index(value[1].default_value)}"/>\n'
                        param_file += f'{tab}\t\t\t</child>\n'
                        param_file += f'{tab}\t\t</choice>\n'
                    elif isinstance(value[1], UniformIntegerHyperparameter):
                        if value[1].lower < -2147483647:
                            lower = -2147483647
                        else:
                            lower = value[1].lower
                        if value[1].upper > 2147483647:
                            upper = 2147483647
                        else:
                            upper = value[1].upper
                        param_file += f'{tab}\t\t\t\t<domain xsi:type=' + \
                            f'"discrete" start="{lower}" end="{upper}" ' + \
                            'defaultIndexOrValue=' + \
                            f'"{value[1].default_value}"/>\n'
                        param_file += f'{tab}\t\t\t</child>\n'
                        param_file += f'{tab}\t\t</choice>\n'
                    elif isinstance(value[1], UniformFloatHyperparameter):
                        param_file += f'{tab}\t\t\t\t<domain xsi:type=' + \
                            f'"continuous" start="{value[1].lower}" end=' + \
                            f'"{value[1].upper}" defaultIndexOrValue=' + \
                            f'"{value[1].default_value}"/>\n'
                        param_file += f'{tab}\t\t\t</child>\n'
                        param_file += f'{tab}\t\t</choice>\n'

                    to_set.remove(child)              

            return param_file, to_set

        for param in hyperparameters:
            if param.name in to_set:
                if param.name in parents and \
                        parents[param.name].keys() in to_set:
                    param_file += f'\t<node xsi:type="or" id="{param.name}">\n'
                    if isinstance(param, CategoricalHyperparameter):
                        choices = ''
                        for c in param.choices:
                            choices += f'{c} '
                        param_file += \
                            '\t\t<domain xsi:type="categorical" strings=' + \
                            f'"{choices[:-1]}" defaultIndexOrValue=' + \
                            f'"{param.choices.index(param.default_value)}"/>\n'

                        children = parents[param.name]
                        param_file, to_set = \
                            set_conditionals(children, param_file, to_set,
                                             parents)
                        param_file += '\t</node>\n'                       

                    elif isinstance(param, UniformIntegerHyperparameter):
                        if param.lower < -2147483647:
                            lower = -2147483647
                        else:
                            lower = param.lower
                        if param.upper > 2147483647:
                            upper = 2147483647
                        else:
                            upper = param.upper
                        param_file += \
                            '  <domain xsi:type="discrete" start=' + \
                            f'"{lower}" end="{upper}" defaultIndexOrValue=' + \
                            f'"{param.default_value}"/>\n'

                        children = parents[param.name]
                        param_file, to_set = \
                            set_conditionals(children, param_file, to_set,
                                             parents)
                        param_file += '\t</node>\n'

                    elif isinstance(param, UniformFloatHyperparameter):
                        param_file += \
                            '\t\t<domain xsi:type="continuous" start=' + \
                            f'"{param.lower}" end="{param.upper}" ' + \
                            f'defaultIndexOrValue="{param.default_value}"/>\n'
                        
                        children = parents[param.name]
                        param_file, to_set = \
                            set_conditionals(children, param_file, to_set,
                                             parents)
                        param_file += '\t</node>\n'
                else:
                    param_file += \
                        f'\t<node xsi:type="value" id="{param.name}">\n'
                    if isinstance(param, CategoricalHyperparameter):
                        choices = ''
                        for c in param.choices:
                            choices += f'{c} '
                        param_file += \
                            '\t\t<domain xsi:type="categorical" strings=' + \
                            f'"{choices[:-1]}" defaultIndexOrValue=' + \
                            f'"{param.choices.index(param.default_value)}"/>\n'
                        param_file += '\t</node>\n'
                    elif isinstance(param, UniformIntegerHyperparameter):
                        if param.lower < -2147483647:
                            lower = -2147483647
                        else:
                            lower = param.lower
                        if param.upper > 2147483647:
                            upper = 2147483647
                        else:
                            upper = param.upper
                        param_file += \
                            '\t\t<domain xsi:type="discrete" start=' + \
                            f'"{lower}" end="{upper}" defaultIndexOrValue=' + \
                            f'"{param.default_value}"/>\n'
                        param_file += '\t</node>\n'
                    elif isinstance(param, UniformFloatHyperparameter):
                        param_file += '\t\t<domain xsi:type="continuous"' + \
                            f' start="{param.lower}" end="{param.upper}"' + \
                            f' defaultIndexOrValue="{param.default_value}"/>\n'
                        param_file += '\t</node>\n'

                to_set.remove(param.name)

        param_file += '</node>\n'

        return param_file
