#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.
pip install -e .
pytest
