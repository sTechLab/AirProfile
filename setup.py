import codecs

from setuptools import setup

with codecs.open('README.rst', 'r', 'utf-8') as fd:
    setup(
        name='AirProfile',
        version='1.0.12',
        description='Automatic analysis of Airbnb host profiles.',
        long_description=fd.read(),
        author='Cornell Tech',
        author_email='kl545@cornell.edu',
        maintainer='Kenneth Lim',
        maintainer_email='kl545@cornell.edu',
        keywords=[
            'Airbnb', 'self-disclosure', 'trustworthiness', 'sharing economy',
            'social exchange'
        ],
        classifiers=[
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
        ],
        url='https://github.com/sTechLab/AirProfile',
        license='MIT',
        install_requires=[
            'beautifulsoup4', 'cachetools', 'lxml', 'nltk', 'numpy', 'pathlib',
            'scikit-learn ~= 0.18.0', 'textstat', 'ujson'
        ],
        include_package_data=True,
        zip_safe=False)
