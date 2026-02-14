#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.
pip install -e .
python -m pytest
