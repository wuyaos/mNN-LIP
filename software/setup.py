from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = f.readlines()

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='nnlip',
    version='1.0.0',
    classifiers=[
        'Development Status :: 3 - Alpha', 'Environment :: Console',
        'Operating System :: OS Independent'
    ],
    license='GPLv3',
    description='NNLIP: A Python package for Nerual Network modifed lattice inversion potential',
    long_description_content_type="text/markdown",
    long_description=readme,
    author='Feifeng Wu',
    author_email='wufeifeng_hust@163.com',
    url='https://stanfordbshan.github.io/HTMACat-kit/',
    keywords=['pytorch', 'Nerual Network potential'],
    python_requires=">=3.6",
    install_requires=requirements,
    packages=find_packages(),
    include_package_data=True,
)