"""Smac algorithm configuration interface for unified planning."""
from AC_interface import GenericACInterface
from utils.pcs_transform import transform_pcs
import random


class SmacInterface(GenericACInterface):
    """Generic Smac interface."""

    def __init__(self):
        """Initialize Smac interface."""
        GenericACInterface.__init__(self)

    def transform_conf_from_ac(self, engine, configuration, plantype):
        """
        Transform a configuration to the format expected by the planning engines.

        :param str engine: Name of the planning engine.
        :param dict configuration: The configuration with parameter names and values.

        :return: The transformed configuration in the engine's expected format.
        :rtype: dict

        :raises ValueError: If the provided engine list is empty or contains non-string elements.
        """
        if engine in ('lpg', 'lpg-anytime'):
            if isinstance(configuration, dict):
                config = configuration
            else:
                config = configuration.get_dictionary()
            config = transform_pcs(engine, configuration)
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
            if isinstance(configuration, dict):
                config = configuration
            else:
                config = configuration.get_dictionary()
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

            if isinstance(configuration, dict):
                config = configuration
            else:
                config = configuration.get_dictionary()

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
            if isinstance(configuration, dict):
                config = configuration
            else:
                config = configuration.get_dictionary()

        elif engine in ('enhsp', 'enhsp-any'):
            if isinstance(configuration, dict):
                config = configuration
            else:
                config = configuration.get_dictionary()
            if not isinstance(config, str) and 'params' not in config:
                conf = ''
                for name, value in config.items():
                    conf += '-' + name[0] + ' ' + value + ' '
                config = {'params': conf}

        return config
