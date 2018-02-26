from setuptools import setup, find_packages

setup(
    name='python-ml-training2',
    version='1.4.0',
    packages=find_packages(exclude=['tests']),
    install_requires=[

        'Flask==0.12.2',
        'flask-mongoengine==0.9.5',
        'Flask-PyMongo==0.5.1',
        'Flask-WTF==0.14.2',
        'matplotlib==2.2.0rc1',
        'mongoengine==0.15.0',
        'numpy==1.14.0',
        'pandas==0.22.0',
        'pymongo==3.6.0',
        'scikit-learn==0.19.1',
        'scipy==1.0.0'
    ]
)