from setuptools import find_packages, setup

setup(
    name='drr_framework',
    version='0.2.0',
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
        'PyYAML',
    ],
    entry_points={
        'console_scripts': [
            'drr-reproduce=drr_framework.experiments:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    python_requires='>=3.8',
)
