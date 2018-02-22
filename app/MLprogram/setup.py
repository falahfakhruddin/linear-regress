from setuptools import setup, find_packages

setup(
    name='python-ml-training2',
    version='1.2.0',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'numpy',
        'pymongo',
        'pandas',
        'scipy',
        'scikit-learn',
        'mongoengine',
        'matplotlib',
        ]
    )


