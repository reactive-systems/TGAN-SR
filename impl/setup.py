from setuptools import setup, find_packages

setup(
    name='tgan_sr',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tensorflow>=2.3.0',
        'matplotlib',
        'sympy'
    ],
    python_requires='>=3.6'
)
