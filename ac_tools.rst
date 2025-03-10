.. _ac-tools:

Automatic algorithm configuration tools
=======================================

Irace
-----

Irace (Iterated Race) is an autoconfiguration tool that focuses on tuning the parameters of optimization algorithms.
It employs a racing mechanism to iteratively select the best configurations among a set of candidate configurations based on their performance on a set of problem instances.

Please note that you need to have a working installation of R on your system.
Then install the irace R package via the R console by running:

.. code-block:: R

    install.packages("remotes")
    install.packages("irace", version = "3.5", repos = "https://cloud.r-project.org")


After that you need to leave the R terminal and install the Irace Python package via

.. code-block:: python

    pip install git+https://github.com/DimitriWeiss/up-iracepy@main



The algorithm configuration implementation will then access irace via the python package rpy2.

For further details on Irace refer to the `irace GitHub <https://github.com/cran/irace>`_ and the `python implementation of irace <https://github.com/auto-optimization/iracepy>`_.


.. automodule:: up_ac.Irace_configurator
    :members:
    :exclude-members: TimeoutException

.. automodule:: up_ac.Irace_interface
    :members:


OAT
---

The Optano algorithm tuner (OAT) executes tuning on optimization functions using different algorithms like GGA, GGA++, JADE, and active CMA-ES.
While it is able to run on a single computing node it also supports multiple workers.
Before being able to use OAT first execute the following code after having installed up-ac.

.. code-block:: python

    from up_ac.utils.download_OAT import get_OAT, copy_call_engine_OAT, delete_OAT
    
    get_OAT()
    copy_call_engine_OAT()


The first function generates a directory for OAT, downloads compiled code for OAT and saves it in the up_ac directory. 
The second function moves code to the OAT directory. 
Once you have run these functions, you do not need to run them again, except if you have removed the OAT directory.

To remove the OAT directory run:

.. code-block:: python

    delete_OAT()

For more details on OAT refer to `the documentation <https://docs.optano.com/algorithm.tuner/current/>`_.


.. automodule:: up_ac.OAT_configurator
    :members:
    :exclude-members: TimeoutException

.. automodule:: up_ac.OAT_interface
    :members:
  

SMAC
----

SMAC3 (Sequential Model-Based Algorithm Configuration) optimizes algorithm parameters by employing a model-based approach, specifically Bayesian Optimization, to predict the performance of different configurations.
It then uses an aggressive racing mechanism to efficiently compare configurations and iteratively refine the model, directing the search towards regions of the space where better configurations are likely to be found.
In the autoconfiguration Smac can also make use of instance features to improve the predictions.

You can install it via

.. code-block:: bash

    pip install swig
    pip install smac==2.0.1


For more details on SMAC refer to `the SMAC3 GitHub <https://automl.github.io/SMAC3/main/>`_.


.. automodule:: up_ac.Smac_configurator
    :members:    
    :exclude-members: TimeoutException

.. automodule:: up_ac.Smac_interface
    :members:
    
    
Selector
--------

Selector is an ensemble-based automated algorithm configurator. The current implementation includes functionalities and models from CPPL, GGA and SMAC.


For more details on Selector refer to `the Selector GitHub <https://github.com/dotbielefeld/selector>`_.

You can install Selector via

.. code-block:: bash

    pip install swig
    pip install selector-ac


.. automodule:: up_ac.Selector_configurator
    :members:   
    :exclude-members: TimeoutException

.. automodule:: up_ac.Selector_interface
    :members:
   

.. _genFunc:

Generic configuration tools
---------------------------

Here is an overview of the general functions and attributes used by all automated algorithm configuration tools.

These can be used to create custom automated algorithm configurators not included in this project.

.. automodule:: up_ac.configurators
    :members:

.. automodule:: up_ac.AC_interface
    :members:
