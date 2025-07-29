from setuptools import setup, find_packages

setup(
    name='drr_framework',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'scikit-learn',
        'networkx',
        'pandas',
        'seaborn',
        'jupyter',
        'notebook',
        'pyinform',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
