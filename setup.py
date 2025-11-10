#!/usr/bin/env python3
"""
Setup script for MASS (Metrics Analytics Super System)
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / 'README.md'
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ''

# Read requirements
requirements_file = Path(__file__).parent / 'requirements.txt'
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='mass',
    version='0.1.0',
    description='MASS - Metrics Analytics Super System',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='YDB Team',
    url='https://github.com/ydb-platform/mass',
    packages=find_packages(),
    python_requires='>=3.10,<3.14',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'mass=mass.core.analytics_job:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)

