.. _installation:

Installation Guide
==================

Requirements
------------

Most of the dependencies will be installed with the pip package.
You will however still need an installation of swig and r-base as well as a recent version of java.
Swig and r-base are required for the Smac and Irace autoconfiguration tools and java is needed for the enhsp engine.
The project is developed in Python 3.8.5. It is highly recommended to use a virtual environment in linux.


Planning Engines
----------------

If one or more of the planning engines that are integrated in this algorithm configuration 
extension is not available on your system, you can install it via:

.. code-block:: bash

    pip install --pre unified-planning[<engine_name>]==1.1.0.113.dev1

Currently these planning engines are integrated and confirmed to be working:

 - LPG
 - Fast-Downward
 - SymK
 - ENHSP
 - Tamer
 - Pyperplan

due to the unified-planning framework still being in development.


up_ac
-----

You can install our project via pip:

.. code-block:: bash

    pip install up_ac

It is possible to adjust the configuration space of each engine according to your needs by passing it to the set_scenario() function. 
Read (https://automl.github.io/ConfigSpace/main/) for details on how to define a ConfigSpace.
