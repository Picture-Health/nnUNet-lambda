from setuptools import setup, find_packages

setup(
    name='nnunetv2',
    version='2.5.1',
    description='nnU-Net v2 - Patched for proper packaging',
    author='Rushil Nagabhushan',
    url='https://github.com/Picture-Health/nnUNet-2.5.1',
    packages=find_packages(),
    python_requires='>=3.9',
)
