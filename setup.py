from setuptools import setup, find_packages

# Read the README file for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='cryoevap',
    version='2.0.0',
    packages=find_packages(),
    
    # Project metadata
    description='Simulation suite for the evaporation of cryogenic liquids in storage tanks',

    # Author information
    author='Felipe Huerta, Ignacio Tapia',
    author_email='fnhuerta@uc.cl, iptapia@uc.cl',
    maintainer='Felipe Huerta, Ignacio Tapia',
    maintainer_email='fnhuerta@uc.cl, iptapia@uc.cl',
    
    # Project URLs
    url='https://github.com/Ignacio-TG/CryoEvap-v2',

    # Package requirements
    python_requires='>=3.12',
    install_requires=[
        'numpy>=2.2.3',
        'pandas>=2.2.3',
        'matplotlib>=3.10.1',
        'scipy>=1.15.2',
        'CoolProp>=6.6.0',
    ],
    
    # Keywords for PyPI search
    keywords='cryogenic, evaporation, storage, tanks, LNG, ammonia, simulation',
    
    # License
    license='GPL-3.0',
    license_files=('LICENSE',)
)