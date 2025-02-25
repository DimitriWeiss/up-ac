.. _examples:

Examples
=======================================

If you have Unified Planning, up_ac and the respective AC method installed as described, you should be able to run the codes repective to the installed AC method.

Quality tuning Fast-Downward with SMAC
--------------------------------------

.. code-block:: python

    """Test up AC implementation."""
    import unified_planning as up
    import up_ac
    import sys
    import os

    # Get example PPDL files from up_ac
    path = '/' + os.path.abspath(up_ac.__file__).strip('/__init__.py')

    # You can also provide a list of tuples (instance, domain) if some domains are
    # located in different places
    instances = [f'{path}/test_problems/visit_precedence/problem.pddl',
                 f'{path}/test_problems/counters/problem.pddl',
                 f'{path}/test_problems/depot/problem.pddl',
                 f'{path}/test_problems/miconic/problem.pddl',
                 f'{path}/test_problems/matchcellar/problem.pddl']

    sys.path.insert(0, sys.path[0].rsplit('up-ac', 1)[0] + '/up-ac')

    from up_ac.Smac_configurator import SmacConfigurator
    from up_ac.Smac_interface import SmacInterface

    engines = ['fast-downward']
    metric = 'quality'
    timelimit = 30
    plantype = 'OneshotPlanner'

    # initialize SMAC Algorithm Configuration interface
    sgaci = SmacInterface()

    # A custom pcs is possible
    sgaci.read_engine_pcs(engines, f'{path}/engine_pcs')

    engine = engines[0]

    # Compute pddl instance features
    instance_features = {}
    for instance in instances:
        instance_features[instance] \
            = sgaci.compute_instance_features(
                instance.rsplit('/', 1)[0] + '/domain.pddl',
                instance)

    # Make UP print less
    up.shortcuts.get_environment().credits_stream = None

    if __name__ == '__main__':

        # Initialize algorithm configurator
        SAC = SmacConfigurator()
        SAC.get_instance_features(instance_features)
        SAC.set_training_instance_set(instances)
        SAC.set_test_instance_set(instances)
        
        SAC.set_scenario(engine, sgaci.engine_param_spaces[engine],
                         sgaci, configuration_time=180, n_trials=30,
                         min_budget=2, max_budget=5, crash_cost=10000,
                         planner_timelimit=timelimit, n_workers=4,
                         instance_features=SAC.instance_features,
                         metric=metric)

        SAC_fb_func = SAC.get_feedback_function(sgaci, engine,
                                                metric, plantype)

        # Run algorithm configuration
        incumbent, _ = SAC.optimize(feedback_function=SAC_fb_func)

        # Check configurations performance
        perf = SAC.evaluate(metric, engine, plantype, SAC.incumbent,
                            sgaci, planner_timelimit=timelimit)

        # Save best configuration found
        SAC.save_config('.', incumbent, sgaci, engine, plantype)


Runtime tuning SymK with OAT
----------------------------

.. code-block:: python

    """Test up AC implementation."""
    import sys
    import os
    import up_ac

    path = '/' + os.path.abspath(up_ac.__file__).strip('/__init__.py')
    sys.path.insert(0, sys.path[0].rsplit('up-ac', 1)[0] + '/up-ac')

    # You can also provide a list of tuples (instance, domain) if some domains are
    # located in different places
    from up_ac.OAT_configurator import OATConfigurator
    from up_ac.OAT_interface import OATInterface

    # pddl instance to test with
    instances = [f'{path}/test_problems/miconic/problem.pddl',
                 f'{path}/test_problems/depot/problem.pddl',
                 f'{path}/test_problems/safe_road/problem.pddl']

    engines = ['symk']
    metric = 'runtime'
    timelimit = 30
    plantype = 'OneshotPlanner'

    # Initialize OAT Algorithm Configuration interface
    ogaci = OATInterface()

    # You can pass multiple planning engines
    ogaci.read_engine_pcs(engines, f'{path}/engine_pcs')

    engine = engines[0]


    if __name__ == '__main__':

        # Initialize algorithm configurator
        OAC = OATConfigurator()
        OAC.set_training_instance_set(instances)
        OAC.set_test_instance_set(instances)

        OAC.set_scenario(engine, ogaci.engine_param_spaces[engine],
                         ogaci, configuration_time=300, n_trials=30,
                         crash_cost=10000, planner_timelimit=timelimit,
                         n_workers=4, instance_features=None, popSize=5,
                         metric=metric, evalLimit=1)
        OAC_fb_func = OAC.get_feedback_function(ogaci, engine,
                                                metric, plantype)
        # run algorithm configuration
        incumbent, _ = OAC.optimize(feedback_function=OAC_fb_func)

        # check configurations performance
        perf = OAC.evaluate(metric, engine, plantype, OAC.incumbent,
                            ogaci)

        # save best configuration found
        OAC.save_config('.', OAC.incumbent, ogaci, engine, plantype)


Anytime tuning ENHSP with Irace
-------------------------------

.. code-block:: python

    """Test up AC implementation."""
    import unified_planning as up
    import up_ac
    import os
    import sys

    # make sure test can be run from anywhere
    path = '/' + os.path.abspath(up_ac.__file__).strip('/__init__.py')
    sys.path.insert(0, sys.path[0].rsplit('up-ac', 1)[0] + '/up-ac')

    from up_ac.Irace_configurator import IraceConfigurator
    from up_ac.Irace_interface import IraceInterface

    # You can also provide a list of tuples (instance, domain) if some domains are
    # located in different places
    instances = [f'{path}/test_problems/depot/problem.pddl',
                 f'{path}/test_problems/counters/problem.pddl',
                 f'{path}/test_problems/citycar/problem.pddl',
                 f'{path}/test_problems/sailing/problem.pddl']

    engines = ['enhsp-any']
    metric = 'quality'
    timelimit = 30
    plantype = 'AnytimePlanner'

    # Initialize Irace Algorithm Configuration interface
    igaci = IraceInterface()

    # A custom pcs is possible
    igaci.read_engine_pcs(engines, f'{path}/engine_pcs')

    engine = engines[0]

    # Make UP print less
    up.shortcuts.get_environment().credits_stream = None

    if __name__ == '__main__':

        # Initialize algorithm configurator
        IAC = IraceConfigurator()
        IAC.set_training_instance_set(instances)
        IAC.set_test_instance_set(instances)

        IAC.set_scenario(engine, igaci.engine_param_spaces[engine],
                         igaci, configuration_time=650, n_trials=5,
                         crash_cost=10000, min_budget=2,
                         planner_timelimit=timelimit, n_workers=4,
                         instance_features=None)

        IAC_fb_func = IAC.get_feedback_function(igaci, engine,
                                                metric, plantype)

        # Run algorithm configuration
        incumbent, _ = IAC.optimize(feedback_function=IAC_fb_func)

        # Check configurations performance
        perf = IAC.evaluate(metric, engine, plantype, IAC.incumbent,
                            igaci, planner_timelimit=timelimit)

        # Save best configuration found
        IAC.save_config('.', IAC.incumbent, igaci, engine, plantype)


Anytime tuning LPG with Selector
--------------------------------

.. code-block:: python

    """Test up AC implementation."""
    import unified_planning as up
    import os
    import sys

    from up_ac.Selector_configurator import SelectorConfigurator
    from up_ac.Selector_interface import SelectorInterface
    import up_ac

    path = '/' + os.path.abspath(up_ac.__file__).strip('/__init__.py')

    # You can also provide a list of tuples (instance, domain) if some domains are
    # located in different places
    train_instances = [f'{path}/test_problems/visit_precedence/problem.pddl',
                       f'{path}/test_problems/counters/problem.pddl',
                       f'{path}/test_problems/depot/problem.pddl']

    sys.path.insert(0, sys.path[0].rsplit('up-ac', 1)[0] + '/up-ac')

    # Mock test instance set for this example
    test_instances = train_instances

    engines = ['lpg-anytime']
    metric = 'quality'
    timelimit = 30
    plantype = 'AnytimePlanner'

    # Initialize Selector Algorithm Configuration interface
    selgaci = SelectorInterface()

    # You can pass multiple planning engines
    selgaci.read_engine_pcs(engines, f'{path}/engine_pcs')

    engine = engines[0]

    # Compute pddl instance features
    instance_features = {}
    for instance in train_instances:
        if isinstance(instance, tuple):
            domain = instance[1]
            instance = instance[0]
            instance_features[instance] \
                = selgaci.compute_instance_features(
                    domain,
                    instance)
        else:
            instance_features[instance] \
                = selgaci.compute_instance_features(
                    instance.rsplit('/', 1)[0] + '/domain.pddl',
                    instance)

    # Make UP print less
    up.shortcuts.get_environment().credits_stream = None

    if __name__ == '__main__':

        # Initialize algorithm configurator
        SelAC = SelectorConfigurator()
        SelAC.get_instance_features(instance_features)
        SelAC.set_training_instance_set(train_instances)
        SelAC.set_test_instance_set(test_instances)

        SelAC.set_scenario(engine, selgaci.engine_param_spaces[engine],
                           selgaci, configuration_time=180, tourn_size=2,
                           min_budget=2, max_budget=3, crash_cost=10000,
                           planner_timelimit=timelimit, n_workers=4, 
                           instance_features=SelAC.instance_features,
                           output_dir='enhsp_selector', metric=metric)

        SelAC_fb_func = SelAC.get_feedback_function(selgaci, engine,
                                                    metric, plantype)

        # Test feedback function
        default_config = \
            selgaci.engine_param_spaces[engine].get_default_configuration()

        # Run algorithm configuration
        incumbent, _ = SelAC.optimize(feedback_function=SelAC_fb_func)

        # Sheck configurations performance on test set
        perf = SelAC.evaluate(metric, engine, plantype, SelAC.incumbent,
                              selgaci, planner_timelimit=timelimit)

        # Save best configuration found
        SelAC.save_config('.', SelAC.incumbent, selgaci, engine, plantype)
