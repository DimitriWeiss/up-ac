# Algorithm Configuration for the AIPlan4EU Unified Planning

Use algorithm configuration on several planners within the unified planning framework to enhance their performance. Development is conducted by the Decision and Operation Technologies group from Bielefeld University (https://github.com/DOTBielefeld).

# Installation

At the moment an installation is only possible via 

```
pip install git+https://github.com/DimitriWeiss/up-ac.git
```

# Planning engines

If a planning engine which is integrated in this algorithm configuration extension is not available on your system, you can install it via

```
pip install --pre unified-planning[<engine_name>]
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

There are three methods currently integrated in this implementation. It is possible to integrate further algorithm configuration methods using the classes
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

. For further details refer to (https://automl.github.io/SMAC3/main/).

## Optano Algorithm Tuner (OAT)

To use OAT, run the following two functions after installation of up_ac.

```
up_ac.utils.download_OAT.get_OAT()
up_ac.utils.download_OAT.copy_call_engine_OAT()
```

The first function generates a directory for OAT, downloads compiled code for OAT and saves it in the up_ac directory. The second function moves code to the OAT directory. Once you have run these functions, you do not need to run them again, except if you have removed the OAT directory.

To remove the OAT directory run:

```
up_ac.utils.download_OAT.delete_OAT()
```

For further details on OAT refer to (https://docs.optano.com/algorithm.tuner/current/).

## Irace

In order to use Irace you need to install R on your system. After that you need to install the R package for irace from the R console by:

```
install.packages("irace", repos = "https://cloud.r-project.org")
```

The algorithm configuration implementation will then access irace via the python package rpy2.

For further details on Irace refer to (https://github.com/cran/irace) and the python implementation of irace (https://github.com/auto-optimization/iracepy).

## Selector

Selector can be installed via

```
pip install selector-ac
```

Selector is an ensemble-based automated algorithm configurator and incorporates functionalities and models from CPPL, GGA and SMAC. Since Selector is implemented with a different version of SMAC than the one used for the SMAC configurator in this library, you should use a separate environment/ virtual environment to configure planners with Selector. For further details on Selector refer to (https://github.com/dotbielefeld/selector).

## Acknowledgments

<img src="https://www.aiplan4eu-project.eu/wp-content/uploads/2021/07/euflag.png" width="60" height="40">

This library is being developed for the AIPlan4EU H2020 project (https://aiplan4eu-project.eu) that is funded by the European Commission under grant agreement number 101016442.