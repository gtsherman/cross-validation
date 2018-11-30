# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='retrievable-cv',
    namespace_packages=['retrievable'],
    version='0.1.0',
    description='',
    long_description=readme,
    author='Garrick Sherman',
    author_email='gsherma2@illinois.edu',
    url='https://github.com/gtsherman/cross-validation',
    license=license,
    packages=find_packages(include=['retrievable', 'retrievable.*']),
    entry_points={
        'console_scripts': [
            'run-cv = retrievable.cv.run:main'
        ]
    },
    install_requires=[
        "scipy",
    ],
)
