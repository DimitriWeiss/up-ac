"""Irace algorithm configuration interface for unified planning."""
import pandas as pd
import random
import sys
import os
import subprocess
import importlib
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

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

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from AC_interface import GenericACInterface
from utils.pcs_transform import transform_pcs


class IraceInterface(GenericACInterface):
    """Irace AC interface."""

    def __init__(self):
        """Initialize Irace interface."""
        GenericACInterface.__init__(self)

        if importlib.metadata.version('ConfigSpace') != '0.6.1':
            print('Installing ConfigSpace 0.6.1')
            subprocess.check_call([sys.executable, "-m", "pip", "install",
                                   "ConfigSpace==0.6.1"],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)

            os.execv(sys.executable, ['python'] + sys.argv)

    def transform_conf_from_ac(self, engine, configuration,
                               plantype='OneshotPlanner'):
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
                        search_option = \
                            config['fast_downward_search_config'] + '('
                        greedy = config['greedy']
                        before_shrinking = config['before_shrinking']
                        before_merging = config['before_merging']
                        if (before_shrinking == 'false'
                                and before_merging == 'false'):
                            before_merging = 'true'
                        max_states = int(float(config['max_states']))
                        threshold_before_merge = \
                            config['threshold_before_merge']

                        search_option += 'merge_and_shrink(merge_strategy='
                        search_option += '''merge_precomputed
                            (merge_tree=linear(variable_order='''
                        search_option += 'reverse_level)),shrink_strategy='
                        search_option += \
                            f'shrink_bisimulation(greedy={greedy}),'
                        search_option += 'label_reduction='
                        search_option += \
                            f'exact(before_shrinking={before_shrinking},'
                        search_option += f'before_merging={before_merging}),'
                        search_option += f'max_states={max_states}'

                        if max_states == 'true': 
                            threshold_before_merge += \
                                ',threshold_before_merge=1))'
                        else:
                            search_option += '))'
                    else:
                        search_option = \
                            config['fast_downward_search_config'] + '('
                        search_option += config['astar_h'] + '())'
                elif (config['fast_downward_search_config']
                        in ('lazy_greedy', 'eager_greedy')):
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
                        if (idx < len(h_name)
                                and 'h' + h_name[h_idx][:2] not in used
                                and idx != int(config['nr_preferred'])):
                            prefs += ','
                        elif (']' not in prefs
                                and idx == int(config['nr_preferred'])):
                            if ('h' + h_name[h_idx][:2] in used
                                    and prefs[-1] == ','):
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
                        if (idx < len(h_name)
                                and 'h' + h_name[h_idx][:2] not in used
                                and idx != int(config['nr_preferred'])):
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

        elif engine in ('enhsp', 'enhsp-any'):
            if not isinstance(configuration, str) and 'params' not in configuration:
                config = ''
                for name, value in configuration.items():
                    config += '-' + name[0] + ' ' + value + ' '
                config = {'params': config}
            else:
                config = configuration

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
