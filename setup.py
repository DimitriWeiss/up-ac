#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='up_ac',
      version='0.0.1',
      description='An Algorithm Configuration package for unified-planning.',
      url='https://github.com/DimitriWeiss/up-ac.git',
      author='DW',
      author_email='dimitri-weiss@web.com',
      packages=find_packages(exclude=["*.tests"]),
      package_data={'': ['test_problems/citycar/*',
                              'test_problems/counters/*',
                              'test_problems/depot/*',
                              'test_problems/htn-transport/*',
                              'test_problems/matchcellar/*',
                              'test_problems/miconic/*',
                              'test_problems/robot_fastener/*',
                              'test_problems/safe_road/*',
                              'test_problems/sailing/*',
                              'test_problems/visit_precedence/*',
                              'engine_pcs/*']},
      include_package_data=True,
      install_requires=["unified-planning", "ConfigSpace",
                        "tarski", "pebble", "dill"],
      license='LICENSE.txt',
      )
