.. _engines:

Planning engines
==================

LPG
---
**Support Features:**

- classical and numeric state variables.
- durative actions.
- Supports the Oneshot Planner in both quality and runtime modes.


Fast-Downward
-------------
**Support Features:**

- Classical planning support with full information, non-numeric, deterministic instantaneous actions.
- Supports Oneshot and Anytime Planner engine in both quality and runtime modes.

**Advantages:**

- Integration within the unified_planning library facilitated by the AIPlan4EU project, making it accessible for broader use.
- Recognized as one of the most successful systems for classical planning.

**Disadvantages:**

- Specificity to classical planning scenarios may limit its applicability to other types of planning problems.

EnhSP
-----

**Support Features:**

- Boolean and numeric state variables, actions, processes, and events (PDDL+ language).
- Supports Oneshot Planner in both quality and runtime modes.

**Advantages:**

- Handles disjunctive preconditions and conditional effects without expensive compilations.

**Disadvantages:**

- Optimality only assured for specific PDDL+ fragments (simple numeric planning problems)​

symk 
----

**Support Features:**

- State-of-the-art classical optimal and top-k planner based on symbolic search extending Fast Downward. It can find a single optimal plan or a set of k different best plans with the lowest cost for a given planning task.
- Supported in the Oneshot and Anytime Planners in both quality and runtime modes.

tamer
------

Tamer is a temporal planner that supports temporal action-based problems.

**Support Features:**

- Supported in the Oneshot Planner in the runtime mode.


pyperplan
---------

**Support Features:**

- Classical planning based on different search heuristics.
- Action-based problems with hierarchical typing.
- Supports the STRIPS PDDL fragment without action costs.
- Supported in the Oneshot Planner in the runtime mode.


**Advantages:**

- Lightweight and written in Python.

**Disadvantages:**

- Does not support action costs in the STRIPS PDDL fragment, which could limit its usefulness in certain planning scenarios.
- The default planning algorithm is a blind breadth-first search, which does not scale well, although other heuristic search algorithms are available.
- Limited to specific PDDL fragments, which may not cater to more complex or varied planning needs.
