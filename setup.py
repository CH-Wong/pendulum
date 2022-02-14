#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='pendulum',
    version='0.1',
    description='Pendulum equation simulation',
    author='Chun Heung Wong',
    author_email='w.chunheung@gmail.com',
    packages=find_packages(include=['pendulum', 'pendulum.*']),
    entry_points={
        'console_scripts': [
            'pendulum=pendulum.animation:main',
            ],
        },
    )