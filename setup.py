#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='up_ac',
      version='0.0.1',
      description='An Algorithm Configuration package for unified-planning.',
      url='https://github.com/DimitriWeiss/up-ac.git',
      author='DW',
      author_email='dimitri.weiss@uni-bielefeld.com',
      packages=find_packages(exclude=["*.tests"]),
      package_data={'up-ac': ['test_problems/citycar/*',
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
      install_requires=["unified-planning", "smac", "ConfigSpace",
                        "tarski", "pebble"],
      include_package_data=True,
      license='LICENSE.txt',
      )
