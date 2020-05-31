#!/bin/bash

echo "Generating H_combined samples..."
python3.6 sample_generation.py h_combined "$@"

echo "Generating n-gram samples..."
python3.6 sample_generation.py ngrams "$@"

echo "Generating SBERT samples..."
python3.6 sample_generation.py sbert "$@"

echo "Generating PPD samples..."
python3.6 sample_generation.py ppd "$@"

echo "Generating clickbait samples"
python3.6 sample_generation.py clickbait "$@"

echo "Generating headline popularity samples"
python3.6 sample_generation.py headline "$@"

echo "Generating summarizer samples"
python3.6 sample_generation.py summarizers "$@"