from setuptools import find_packages, setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(
    name='AirProfile',
    version='1.0.9',
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
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False)
