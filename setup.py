"""
Setup script for the Something-Something V2 project.
"""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='something-something-v2-3d',
    version='0.1.0',
    author='Rasmus Arnmark',
    description='3D CNN implementation for Something-Something V2 dataset',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/RasmusArnmark/something-something-IV',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
)
