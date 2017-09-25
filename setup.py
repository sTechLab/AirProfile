from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(
    name='AirProfile',
    version='1.0.12',
    description='Automatic analysis of Airbnb host profiles.',
    long_description=readme(),
    author='Cornell Tech',
    author_email='kl545@cornell.edu',
    maintainer='Kenneth Lim',
    maintainer_email='kl545@cornell.edu',
    keywords=[
        'Airbnb', 'self-disclosure', 'trustworthiness', 'sharing economy',
        'social exchange'
    ],
    url='https://github.com/sTechLab/AirProfile',
    license='MIT',
    install_requires=[
        'beautifulsoup4',
        'cachetools',
        'lxml',
        'nltk',
        'numpy',
        'pathlib',
        'scikit-learn ~= 0.18.0',
        'textstat',
        'ujson'
    ],
    include_package_data=True,
    zip_safe=False)
