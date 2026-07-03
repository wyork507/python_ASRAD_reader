from setuptools import setup, find_packages

setup(
    name="asrad-reader",
    version='0.4.2',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.26.2',
        'pandas>=2.1.4',
    ],
    python_requires='>=3.11.7',
    author="wyork507",
    author_email="wyork507@gmail.com",
    description="A toolkit for processing CWA long-term observation station data from https://asrad.pccu.edu.tw.",
    url="https://github.com/wyork507/python_asrad_reader",
    long_description=open('readme.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
)