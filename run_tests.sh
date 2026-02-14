#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.
python -m pip install -e . --no-build-isolation
python -m pytest
