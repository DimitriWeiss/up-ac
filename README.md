# Algorithm Configuration for the AIPlan4EU Unified Planning

Use algorithm configuration on several planners within the unified planning framework to enhance their performance. Find the documentation at (https://up-ac.readthedocs.io/en/latest/). Development is conducted by the Decision and Operation Technologies group from Bielefeld University (https://github.com/DOTBielefeld).

# Installation

You need to first install Unified Planning before installing up_ac. It is highly recommended to use a separate virtual environment for up_ac.

## Unified Planning

This project was developed in Python 3.8.5. The integrated version of Unified Planning is 1.1.0.113.dev1. The easiest way to install Unified Planning with all available engines is via

```
pip install --pre unified-planning[engines]==1.1.0.113.dev1
```

## up_ac

You can then install up_ac via 

```
pip install up_ac
```

## Planning engines integrated in the algorithm configuration

The development of the unified planning framework is still ongoing. Hence, some of the integrated planning engines are not yet available for automated algorithm configuration. Planning engines confirmed to work in this implementation are:

- LPG
- Fast-Downward
- SymK
- ENHSP
- Tamer
- Pyperplan

It is possible to adjust the configuration space of each engine according to your needs by passing it to the set_scenario() function. Read (https://automl.github.io/ConfigSpace/main/) for details on how to define a ConfigSpace.

# Automated Algorithm Configuration methods

There are four methods currently integrated in this implementation. It is possible to integrate further algorithm configuration methods using the classes
```
up_ac.configurators.Configurator
```
and
```
up_ac.AC_interface.GenericACInterface
```

The methods integrated are:

## SMAC3

Smac can be installed via 

```
pip install smac==2.0.1
```

For further details refer to (https://automl.github.io/SMAC3/main/).

## Optano Algorithm Tuner (OAT)

To use OAT, run the following in Python after installation of up_ac.

```
from up_ac.utils.download_OAT import get_OAT, copy_call_engine_OAT

get_OAT()
copy_call_engine_OAT()
```

The first function generates a directory for OAT, downloads compiled code for OAT and saves it in the up_ac directory. The second function moves code to the OAT directory. Once you have run these functions, you do not need to run them again, except if you have removed the OAT directory.

To remove the OAT directory run:

```
from up_ac.utils.download_OAT import delete_OAT

delete_OAT()
```

For further details on OAT refer to (https://docs.optano.com/algorithm.tuner/current/).

## Irace

In order to use Irace you need to have R on your system. You also need to install the R packages 'remotes' and 'irace' from the R console by:

```
install.packages("remotes")
remotes::install_version("irace", version = "3.5", repos = "https://cloud.r-project.org")
```

After that you need to leave the R terminal and install the Irace Python package via

```
pip install git+https://github.com/DimitriWeiss/up-iracepy@main
```

It is the Irace version implemented in up_ac and secured on this account for that reason. The algorithm configuration implementation will then access irace via the python package rpy2.

For further details on Irace refer to (https://github.com/cran/irace) and the python implementation of irace (https://github.com/auto-optimization/iracepy).

## Selector

Selector can be installed via

```
pip install swig
pip install selector-ac
```

Preinstalling swig may be necessary, depending on your Python interpreter. Selector is an ensemble-based automated algorithm configurator and incorporates functionalities and models from CPPL, GGA and SMAC. Since Selector is implemented with a different version of SMAC than the one used for the SMAC configurator in this library, you should use a separate environment/ virtual environment to configure planners with Selector. For further details on Selector refer to (https://github.com/dotbielefeld/selector).

## Acknowledgments

<img src="https://www.aiplan4eu-project.eu/wp-content/uploads/2021/07/euflag.png" width="60" height="40">

This library is being developed for the AIPlan4EU H2020 project (https://aiplan4eu-project.eu) that is funded by the European Commission under grant agreement number 101016442.