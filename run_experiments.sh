#!/bin/bash


echo "Running hand-crafted feature experiments..."
python3.6 experiments_handcrafted.py "$@"

echo "Running n-gram feature experiments..."
python3.6 experiments_ngrams.py "$@"

echo "Running SBERT progression experiments..."
python3.6 experiments_progression.py "$@"

echo "Running cross-task experiments"
python3.6 experiments_cross_task.py "$@"
