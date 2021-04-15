#! /usr/bin/env python
"""turbofan setup file."""
import pathlib
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import versioneer

# The directory containing this file
HERE = pathlib.Path(__file__).parent

DESCRIPTION = 'Turbofan experiments'
LONG_DESCRIPTION = HERE.joinpath('README.md').read_text()

DISTNAME = 'turbofan'
LICENSE = 'MIT'
AUTHOR = 'Axel Fahy'
EMAIL = 'axel@fahy.net'
URL = 'https://github.com/axelfahy/Turbofan'
DOWNLOAD_URL = ''
PROJECT_URLS = {
    'Bug Tracker': 'https://github.com/axelfahy/Turbofan/issues',
    'Documentation': 'https://github.com/axelfahy/Turbofan',
    'Source Code': 'https://github.com/axelfahy/Turbofan'
}
REQUIRES = [
    'click==4.0.0',
    'click-pathlib',
    'bff@git+https://github.com/axelfahy/bff',
    'defusedxml==0.6.0',
    'dvc>=2.0.17',
    'loguru==0.5.3',
    'pandas==1.2.4',
    'pyarrow==3.0.0',
    'sklearn==0.0',
    'tensorflow-gpu==2.5.0rc1',
    'tqdm==4.60.0',
    'xgboost==1.4.0',
]
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8']


class NoopTestCommand(TestCommand):
    def __init__(self, dist):
        print('Bff does not support running tests with '
              '`python setup.py test`. Please run `make all`.')


cmdclass = versioneer.get_cmdclass()
cmdclass.update({"test": NoopTestCommand})

setup(name=DISTNAME,
      maintainer=AUTHOR,
      version=versioneer.get_version(),
      packages=find_packages(exclude=('tests',)),
      maintainer_email=EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      cmdclass=cmdclass,
      url=URL,
      download_url=DOWNLOAD_URL,
      project_urls=PROJECT_URLS,
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      classifiers=CLASSIFIERS,
      python_requires='>=3.8',
      install_requires=REQUIRES,
      zip_safe=False)
