from setuptools import setup, find_packages

setup(
    name='version-3-multi-agent',
    version='0.1',
    packages=find_packages(include=['agents', 'app', 'client', 'models', 'server', 'utilities', 'agents.*', 'app.*', 'client.*', 'models.*', 'server.*', 'utilities.*']),
    install_requires=[],
)