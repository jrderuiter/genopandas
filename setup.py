#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

REQUIREMENTS = [
    'intervaltree', 'pandas', 'numpy', 'natsort'
]

setup(
    name='genopandas',
    version='0.0.1',
    description='',
    author='Julian de Ruiter',
    author_email='julianderuiter@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=REQUIREMENTS,
    license="MIT license",
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
    ]
)
