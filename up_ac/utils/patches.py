"""This module includes the necessary patches."""
import pyparsing
from ConfigSpace.read_and_write.pcs import (
    ConfigurationSpace,
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    CategoricalHyperparameter,
    pp_cont_param,
    pp_condition,
    pp_cat_param,
    pp_forbidden_clause,
    ForbiddenEqualsClause,
    ForbiddenAndConjunction,
    OrderedDict,
    EqualsCondition,
    InCondition,
    AndConjunction
)


def patch_pcs(pcs):
    """Patch pcs.

    parameter pcs: module.
    
    return pcs (patched module)
    """
    pcs.pp_param_name = pyparsing.Word(
        pyparsing.alphanums + "_" + "-" + "@" + "." + ":" + ";" + "\\" + 
        "=" + "/" + "?" + "!" + "$" + "%" + "&" + "*" + "+" + "<" + ">" + 
        "("  "((" + ")" + "))")
    pcs.pp_choices = pcs.pp_param_name + \
        pyparsing.Optional(pyparsing.OneOrMore("," + pcs.pp_param_name))
    pcs.pp_cat_param = pcs.pp_param_name + "{" + pcs.pp_choices + "}" + \
        "[" + pcs.pp_param_name + "]"

    def _new_read(pcs_string) -> ConfigurationSpace:
        """
        Read in a 
        :py:class:`~ConfigSpace.configuration_space.ConfigurationSpace`
        definition from a pcs file.


        .. code:: python

            from ConfigSpace import ConfigurationSpace
            from ConfigSpace.read_and_write import pcs

            cs = ConfigurationSpace({"a": [1, 2, 3]})
            with open('configspace.pcs', 'w') as f:
                 f.write(pcs.write(cs))

            with open('configspace.pcs', 'r') as f:
                deserialized_conf = pcs.read(f)

        Parameters
        ----------
        pcs_string : Iterable[str]
            ConfigSpace definition in pcs format as an iterable of strings

        Returns
        -------
        :py:class:`~ConfigSpace.configuration_space.ConfigurationSpace`
            The deserialized ConfigurationSpace object
        """
        if isinstance(pcs_string, str):
            pcs_string = pcs_string.split("\n")

        configuration_space = ConfigurationSpace()
        conditions = []
        forbidden = []

        # some statistics
        ct = 0
        cont_ct = 0
        cat_ct = 0
        line_ct = 0

        for line in pcs_string:
            line_ct += 1

            if "#" in line:
                # It contains a comment
                pos = line.find("#")
                line = line[:pos]

            # Remove quotes and whitespaces at beginning and end
            line = line.replace('"', "").replace("'", "")
            line = line.strip()

            if "|" in line:
                # It's a condition
                try:
                    c = pp_condition.parseString(line)
                    conditions.append(c)
                except pyparsing.ParseException as e:
                    raise NotImplementedError(
                        f"Could not parse condition: {line}") from e

                continue
            if "}" not in line and "]" not in line:
                continue
            if line.startswith("{") and line.endswith("}"):
                forbidden.append(line)
                continue
            if len(line.strip()) == 0:
                continue

            ct += 1
            param = None

            create = {
                "int": UniformIntegerHyperparameter,
                "float": UniformFloatHyperparameter,
                "categorical": CategoricalHyperparameter,
            }

            try:
                param_list = pp_cont_param.parseString(line)
                il = param_list[9:]
                if len(il) > 0:
                    il = il[0]
                param_list = param_list[:9]
                name = param_list[0]
                lower = float(param_list[2])  # type: ignore
                upper = float(param_list[4])  # type: ignore
                paramtype = "int" if "i" in il else "float"
                log = "l" in il
                default_value = float(param_list[7])  # type: ignore
                param = create[paramtype](
                    name=name,
                    lower=lower,
                    upper=upper,
                    q=None,
                    log=log,
                    default_value=default_value,
                )
                cont_ct += 1
            except pyparsing.ParseException:
                pass

            try:
                param_list = pp_cat_param.parseString(line)
                name = param_list[0]
                choices = list(param_list[2:-4:2])
                default_value = param_list[-2]
                param = create["categorical"](
                    name=name, choices=choices, default_value=default_value)
                cat_ct += 1
            except pyparsing.ParseException:
                pass

            if param is None:
                raise NotImplementedError("Could not parse: %s" % line)

            configuration_space.add_hyperparameter(param)

        for clause in forbidden:
            # TODO test this properly!
            # TODO Add a try/catch here!
            # noinspection PyUnusedLocal
            param_list = pp_forbidden_clause.parseString(clause)
            tmp_list: list = []
            clause_list = []
            for value in param_list[1:]:
                if len(tmp_list) < 3:
                    tmp_list.append(value)
                else:
                    # So far, only equals is supported by SMAC
                    if tmp_list[1] == "=":
                        # TODO maybe add a check if the hyperparameter is
                        # actually in the configuration space
                        if isinstance(configuration_space[tmp_list[0]],
                                      UniformIntegerHyperparameter):
                            vc_val = int(tmp_list[2])
                        elif isinstance(configuration_space[tmp_list[0]],
                                        UniformFloatHyperparameter):
                            vc_val = float(tmp_list[2])
                        else:
                            vc_val = tmp_list[2]
                        clause_list.append(
                            ForbiddenEqualsClause(
                                configuration_space[tmp_list[0]], vc_val),
                        )
                    else:
                        raise NotImplementedError()
                    tmp_list = []
            configuration_space.add_forbidden_clause(
                ForbiddenAndConjunction(*clause_list))

        # Now handle conditions
        # If there are two conditions for one child, these two conditions 
        # are an AND-conjunction of conditions, thus we have to connect them
        conditions_per_child: dict = OrderedDict()
        for condition in conditions:
            child_name = condition[0]
            if child_name not in conditions_per_child:
                conditions_per_child[child_name] = []
            conditions_per_child[child_name].append(condition)

        for child_name in conditions_per_child:
            condition_objects = []
            for condition in conditions_per_child[child_name]:
                child = configuration_space[child_name]
                parent_name = condition[2]
                parent = configuration_space[parent_name]
                restrictions = condition[5:-1:2]

                # TODO: cast the type of the restriction!
                if len(restrictions) == 1:
                    if isinstance(parent, UniformIntegerHyperparameter):
                        restrictions[0] = int(restrictions[0])
                    elif isinstance(parent, UniformFloatHyperparameter):
                        restrictions[0] = float(restrictions[0])
                    condition = EqualsCondition(child, parent, restrictions[0])
                else:
                    if isinstance(parent, UniformIntegerHyperparameter):
                        for i, r in enumerate(restrictions):
                            restrictions[i] = int(r)
                    elif isinstance(parent, UniformFloatHyperparameter):
                        for i, r in enumerate(restrictions):
                            restrictions[i] = float(r)
                    condition = InCondition(child, parent, values=restrictions)
                condition_objects.append(condition)

            # Now we have all condition objects for this child,
            # so we can build a giant AND-conjunction of them
            # (if number of conditions >= 2)!

            if len(condition_objects) > 1:
                and_conjunction = AndConjunction(*condition_objects)
                configuration_space.add_condition(and_conjunction)
            else:
                configuration_space.add_condition(condition_objects[0])

        return configuration_space

    pcs.read = _new_read

    return pcs
