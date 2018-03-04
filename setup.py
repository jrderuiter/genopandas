#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    README = readme_file.read()

with open('HISTORY.rst') as history_file:
    HISTORY = history_file.read()

REQUIREMENTS = ['intervaltree', 'pandas', 'numpy', 'natsort', 'toolz']


setup(
    name='genopandas',
    description=('Datastructures for manipulating, querying '
                 'and plotting genomic data.'),
    long_description=README + '\n\n' + HISTORY,
    author='Julian de Ruiter',
    author_email='julianderuiter@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=REQUIREMENTS,
    setup_requires=['setuptools_scm'],
    use_scm_version={'root': '.', 'relative_to': __file__},
    license='MIT license',
    url='https://github.com/jrderuiter/genopandas',
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
    ])
