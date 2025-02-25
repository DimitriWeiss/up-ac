"""OAT algorithm configuration interface for unified planning."""

from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.conditions import (
    AndConjunction
)
import random
import sys
import os
import subprocess
import importlib

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from AC_interface import GenericACInterface
from utils.pcs_transform import transform_pcs


class OATInterface(GenericACInterface):
    """
    OAT AC interface.

    OAT does not handle forbidden parameter value combinations.
    OAT can handle multiple parent and 1 child conditionals,
    but not one parent multiple children conditionals.
    We naively just take the first one in the list.
    OAT does not support conditionals that are conditional.
    We leave them out naively.
    OAT does not support conditionals with value ranges.
    We naively only use the first value.

    Note: Although this is suboptimal, invalid configurations will
    lead to crash or bad results such that OAT will rate them
    as subpar.
    
    """

    def __init__(self):
        """Initialize OAT interface."""
        GenericACInterface.__init__(self)

        if importlib.metadata.version('ConfigSpace') != '0.6.1':
            print('Installing ConfigSpace 0.6.1')
            subprocess.check_call([sys.executable, "-m", "pip", "install",
                                   "ConfigSpace==0.6.1"],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)

            os.execv(sys.executable, ['python'] + sys.argv)

    def transform_conf_from_ac(self, engine, configuration, plantype):
        """
        Transform a configuration to the UP engine format.

        :param engine: The name of the engine.
        :type engine: str
        :param configuration: Parameter names with values.
        :type configuration: dict

        :return: The transformed configuration.
        :rtype: dict
        """
        config = transform_pcs(engine, configuration)
        if engine in ('lpg', 'lpg-anytime'):
            del_list = []
            add_list = []
            for pname, pvalue in config.items():
                if pname in self.engine_param_types[engine]:
                    if self.engine_param_types[engine][pname] == 'FLAGS':
                        del_list.append(pname)
                        flag_pname = pname + '=' + config[pname]
                        add_list.append(flag_pname)
                    elif self.engine_param_types[engine][pname] == 'FLAG':
                        if config[pname] == '1':
                            config[pname] = ''
                        else:
                            del_list.append(pname)

            for dl in del_list:
                del config[dl]

            for al in add_list:
                config[al] = ''

        elif engine == 'fast-downward':
            if len(config) == 1:
                pass
            else:
                if config['fast_downward_search_config'] == 'astar':
                     
                    if config['astar_h'] == 'merge_and_shrink':
                        search_option = config['fast_downward_search_config'] + '('
                        greedy = config['greedy']
                        before_shrinking = config['before_shrinking']
                        before_merging = config['before_merging']
                        if before_shrinking == 'false' and before_merging == 'false':
                            before_merging = 'true'
                        max_states = int(float(config['max_states']))
                        threshold_before_merge = config['threshold_before_merge']

                        search_option += 'merge_and_shrink(merge_strategy='
                        search_option += 'merge_precomputed(merge_tree=linear(variable_order='
                        search_option += 'reverse_level)),shrink_strategy='
                        search_option += f'shrink_bisimulation(greedy={greedy}),'
                        search_option += 'label_reduction='
                        search_option += f'exact(before_shrinking={before_shrinking},'
                        search_option += f'before_merging={before_merging}),'
                        search_option += f'max_states={max_states}'

                        if max_states == 'true': 
                            threshold_before_merge += ',threshold_before_merge=1))'
                        else:
                            search_option += '))'
                    else:
                        search_option = config['fast_downward_search_config'] + '('
                        search_option += config['astar_h'] + '())'
                elif config['fast_downward_search_config'] in ('lazy_greedy', 'eager_greedy'):
                    h_def = ''
                    h_name = []
                    search_option = ''
                    skip = 0
                    for i in range(int(config['number_h'])):
                        idx = int(i) + 1
                        h = config[f'h_{idx}']
                        if h in h_name:
                            skip += 1
                            continue
                        h_name.append(h)
                        if idx > len(h_name):
                            idxxx = idx - skip
                        else:
                            idxxx = idx
                        h_def += f'let(h{h_name[idxxx - 1][:2]},{h}(),'
                    search_option += h_def
                    search_option += config['fast_downward_search_config']
                    search_option += '(['
                    for i, hn in enumerate(h_name):
                        search_option += 'h' + hn[:2]
                        if i + 1 < len(h_name):
                            search_option += ','
                        else:
                            search_option += '],'
                    search_option += 'preferred=['
                    prefs = ''
                    used = []
                    for i in range(int(config['nr_preferred'])):
                        idx = int(i) + 1
                        if idx <= int(config['number_h']):
                            h_idx = int(float(config[f'preferred_{idx}'])) - 1
                            if h_idx > len(h_name) - 1:
                                if len(h_name) > 1:
                                    h_idx = random.randint(0, len(h_name) - 1)
                                else:
                                    h_idx = 0
                            if 'h' + h_name[h_idx][:2] not in used not in used:
                                prefs += 'h' + h_name[h_idx][:2]
                        if idx < len(h_name) and 'h' + h_name[h_idx][:2] not in used and idx != int(config['nr_preferred']):
                            prefs += ','
                        elif ']' not in prefs and idx == int(config['nr_preferred']):
                            if 'h' + h_name[h_idx][:2] in used and prefs[-1] == ',':
                                prefs = prefs[:-1]
                            prefs += '],'
                        used.append('h' + h_name[h_idx][:2])
                    search_option += prefs
                    cost_type = config['cost_type']
                    search_option += f'cost_type={cost_type}))'
                    for _ in range(len(h_name) - 1):
                        search_option += ')'

                elif config['fast_downward_search_config'] == 'lazy_wastar':
                    h_def = ''
                    h_name = []
                    search_option = ''
                    skip = 0
                    for i in range(int(config['number_h'])):
                        idx = int(i) + 1
                        h = config[f'h_{idx}']
                        if h in h_name:
                            skip += 1
                            continue
                        h_name.append(h)
                        if idx > len(h_name):
                            idxxx = idx - skip
                        else:
                            idxxx = idx
                        h_def += f'let(h{h_name[idxxx - 1][:2]},{h}(),'
                    search_option += h_def
                    search_option += config['fast_downward_search_config']
                    search_option += '(['
                    for i, hn in enumerate(h_name):
                        search_option += 'h' + hn[:2]
                        if i + 1 < len(h_name):
                            search_option += ','
                        else:
                            search_option += '],'
                    w = int(float(config['w']))
                    search_option += f'w={w},'
                    search_option += 'preferred=['
                    prefs = ''
                    used = []
                    for i in range(int(config['nr_preferred'])):
                        idx = int(i) + 1    
                        if idx <= int(config['number_h']):
                            h_idx = int(float(config[f'preferred_{idx}'])) - 1
                            if h_idx > len(h_name) - 1:
                                if len(h_name) > 1:
                                    h_idx = random.randint(0, len(h_name) - 1)
                                else:
                                    h_idx = 0
                            if 'h' + h_name[h_idx][:2] not in used not in used:
                                prefs += 'h' + h_name[h_idx][:2]
                        if idx < len(h_name) and 'h' + h_name[h_idx][:2] not in used and idx != int(config['nr_preferred']):
                            prefs += ','
                        elif ']' not in prefs and idx == int(config['nr_preferred']):
                            if 'h' + h_name[h_idx][:2] in used and prefs[-1] == ',':
                                prefs = prefs[:-1]
                            prefs += '],'
                        used.append('h' + h_name[h_idx][:2])
                    search_option += prefs
                    cost_type = config['cost_type']
                    search_option += f'cost_type={cost_type}))'
                    for _ in range(len(h_name) - 1):
                        search_option += ')'

                elif config['fast_downward_search_config'] == 'eager':
                    h = config['h_1']
                    h_name = h
                    search_option = ''
                    h_def = f'let(h{h_name[:2]},{h}(),'
                    weight = int(float(config['weight']))
                    search_option += h_def
                    search_option += config['fast_downward_search_config']
                    search_option += '(single(sum([g(),weight(' + f'h{h_name[:2]}'
                    search_option += f',{weight})])),'
                    search_option += f'preferred=[h{h_name[:2]}],'
                    cost_type = config['cost_type']
                    search_option += f'cost_type={cost_type}))'

                if plantype == 'OneshotPlanner':
                    config = {'fast_downward_search_config': search_option}

                elif plantype == 'AnytimePlanner':
                    config = {'fast_downward_anytime_search_config': search_option}

        elif engine == 'symk':

            if len(config) == 1:
                pass
            else:
                if config['fast_downward_search_config'] == 'astar':
                     
                    if config['astar_h'] == 'merge_and_shrink':
                        search_option = config['fast_downward_search_config'] + '('
                        greedy = config['greedy']
                        before_shrinking = config['before_shrinking']
                        before_merging = config['before_merging']
                        if before_shrinking == 'false' and before_merging == 'false':
                            before_merging = 'true'
                        max_states = int(float(config['max_states']))
                        threshold_before_merge = config['threshold_before_merge']

                        search_option += 'merge_and_shrink(merge_strategy='
                        search_option += 'merge_precomputed(merge_tree=linear(variable_order='
                        search_option += 'reverse_level)),shrink_strategy='
                        search_option += f'shrink_bisimulation(greedy={greedy}),'
                        search_option += 'label_reduction='
                        search_option += f'exact(before_shrinking={before_shrinking},'
                        search_option += f'before_merging={before_merging}),'
                        search_option += f'max_states={max_states}'

                        if max_states == 'true': 
                            threshold_before_merge += ',threshold_before_merge=1))'
                        else:
                            search_option += '))'
                    else:
                        search_option = config['fast_downward_search_config'] + '('
                        search_option += config['astar_h'] + '())'

                elif config['fast_downward_search_config'] in ('lazy_greedy', 'eager_greedy'):
                    h_def = ''
                    h_name = []
                    search_option = ''
                    skip = 0
                    for i in range(int(config['number_h'])):
                        idx = int(i) + 1
                        h = config[f'h_{idx}'] + '()'
                        if h in h_name:
                            skip += 1
                            continue
                        h_name.append(h)
                    search_option += config['fast_downward_search_config']
                    search_option += '(['
                    for i, hn in enumerate(h_name):
                        search_option += hn
                        if i + 1 < len(h_name):
                            search_option += ','
                        else:
                            search_option += '],'
                    search_option += 'preferred=['
                    prefs = ''
                    used = []
                    for i in range(int(config['nr_preferred'])):
                        idx = int(i) + 1
                        if idx <= int(config['number_h']):
                            h_idx = int(float(config[f'preferred_{idx}'])) - 1
                            if h_idx > len(h_name) - 1:
                                if len(h_name) > 1:
                                    h_idx = random.randint(0, len(h_name) - 1)
                                else:
                                    h_idx = 0
                            if h_name[h_idx] not in used:
                                prefs += h_name[h_idx]
                        if idx < len(h_name) and h_name[h_idx] not in used and idx != int(config['nr_preferred']):
                            prefs += ','
                        elif ']' not in prefs and idx == int(config['nr_preferred']):
                            if h_name[h_idx] in used and prefs[-1] == ',':
                                prefs = prefs[:-1]
                            prefs += '],'
                        used.append(h_name[h_idx])
                    search_option += prefs
                    cost_type = config['cost_type']
                    search_option += f'cost_type={cost_type})'

                elif config['fast_downward_search_config'] == 'lazy_wastar':
                    h_def = ''
                    h_name = []
                    search_option = ''
                    skip = 0
                    for i in range(int(config['number_h'])):
                        idx = int(i) + 1
                        h = config[f'h_{idx}'] + '()'
                        if h in h_name:
                            skip += 1
                            continue
                        h_name.append(h)
                    search_option += config['fast_downward_search_config']
                    search_option += '(['
                    for i, hn in enumerate(h_name):
                        search_option += hn
                        if i + 1 < len(h_name):
                            search_option += ','
                        else:
                            search_option += '],'
                    w = int(float(config['w']))
                    search_option += f'w={w},'
                    search_option += 'preferred=['
                    prefs = ''
                    used = []
                    for i in range(int(config['nr_preferred'])):
                        idx = int(i) + 1    
                        if idx <= int(config['number_h']):
                            h_idx = int(float(config[f'preferred_{idx}'])) - 1
                            if h_idx > len(h_name) - 1:
                                if len(h_name) > 1:
                                    h_idx = random.randint(0, len(h_name) - 1)
                                else:
                                    h_idx = 0
                            if h_name[h_idx] not in used not in used:
                                prefs += h_name[h_idx]
                        if idx < len(h_name) and h_name[h_idx] not in used and idx != int(config['nr_preferred']):
                            prefs += ','
                        elif ']' not in prefs and idx == int(config['nr_preferred']):
                            if h_name[h_idx] in used and prefs[-1] == ',':
                                prefs = prefs[:-1]
                            prefs += '],'
                        used.append(h_name[h_idx])
                    search_option += prefs
                    cost_type = config['cost_type']
                    search_option += f'cost_type={cost_type})'

                elif config['fast_downward_search_config'] == 'eager':
                    h = config['h_1'] + '()'
                    h_name = h
                    search_option = ''
                    weight = int(float(config['weight']))
                    search_option += config['fast_downward_search_config']
                    search_option += '(single(sum([g(),weight(' + f'{h_name}'
                    search_option += f',{weight})])),'
                    search_option += f'preferred=[{h_name}],'
                    cost_type = config['cost_type']
                    search_option += f'cost_type={cost_type})'

                if plantype == 'OneshotPlanner':
                    config = {'symk_search_config': search_option}

                elif plantype == 'AnytimePlanner':
                    config = {'symk_anytime_search_config': search_option}

        elif engine in ('tamer', 'pyperplan'):
            config = {}
            for param in \
                    self.engine_param_spaces[engine].get_hyperparameters():
                if isinstance(param, UniformFloatHyperparameter):
                    config[param.name] = float(configuration[param.name])
                elif isinstance(param, UniformIntegerHyperparameter):
                    config[param.name] = int(configuration[param.name])
                else:
                    config[param.name] = configuration[param.name]

        elif engine in ('enhsp', 'enhsp-any'):
            if not isinstance(configuration, str) and 'params' not in configuration:
                config = ''
                for name, value in configuration.items():
                    config += '-' + name[0] + ' ' + value + ' '
                config = {'params': config}
            else:
                config = configuration
 
        return config

    def get_ps_oat(self, param_space):
        """
        Generate the OAT parameter tree in XML format.

        OAT does not handle forbidden parameter value combinations.
        OAT can handle multiple parent and 1 child conditionals,
        but not one parent multiple children conditionals.
        We naively just take the first one in the list.
        OAT does not support conditionals that are conditional.
        We leave them out naively.
        OAT does not support conditionals with value ranges.
        We naively only use the first value.

        Note: Although this is suboptimal, invalid configurations will
        lead to crash or bad results such that OAT will rate them
        as subpar.

        :param param_space: ConfigSpace object.
        :type param_space: ConfigSpace.ConfigurationSpace

        :return: OAT parameter tree in XML format.
        :rtype: str

        """

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

            :param children: Child parameters.
            :type children: dict
            :param param_file: OAT parameter tree to be saved in XML.
            :type param_file: str
            :param to_set: Parameter names to still be included.
            :type to_set: list
            :param parents: Parent parameters.
            :type parents: dict
            :param tab: Indicates depth of tree (\t).
            :type tab: str

            :return: Updated OAT parameter tree and parameter names to still be included.
            :rtype: tuple
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
