"""Smac algorithm configuration interface for unified planning."""
from up_ac.AC_interface import GenericACInterface
from up_ac.utils.pcs_transform import transform_pcs


class SmacInterface(GenericACInterface):
    """Generic Smac interface."""

    def __init__(self):
        """Initialize Smac interface."""
        GenericACInterface.__init__(self)

    def transform_conf_from_ac(self, engine, configuration):
        """
        Transform a configuration to the format expected by the planning engines.

        :param str engine: Name of the planning engine.
        :param dict configuration: The configuration with parameter names and values.

        :return: The transformed configuration in the engine's expected format.
        :rtype: dict

        :raises ValueError: If the provided engine list is empty or contains non-string elements.
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

        elif engine in ('enhsp', 'tamer', 'pyperplan'):
            if isinstance(configuration, dict):
                config = configuration
            else:
                config = configuration.get_dictionary()

        return config
