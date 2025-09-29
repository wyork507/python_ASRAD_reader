from setuptools import setup, find_packages

setup(
    name="ASRAD_reader",
    version='0.4.1',
    packages=find_packages(),
    install_requires=[
        'numpy  >= 1.26.2',
        'pandas >= 2.1.4',
        'python >= 3.11.7',
    ],
)