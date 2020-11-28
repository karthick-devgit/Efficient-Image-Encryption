from setuptools import find_packages, setup

setup(
    name='secureimage',
    packages=find_packages(),
    version='0.1.0',
    description='A python library to encrypt and decrypt images securely and efficiently',
    author='Karthick S and Sandhya G',
    license='MIT',
    install_requires=['numpy','Pillow'],
)
