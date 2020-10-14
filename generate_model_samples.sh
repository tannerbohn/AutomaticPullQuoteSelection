#!/bin/bash

echo "Generating handcrafted samples..."
python3.6 sample_generation.py handcrafted "$@"

echo "Generating n-gram samples..."
python3.6 sample_generation.py ngrams "$@"

echo "Generating SBERT C-deep samples..."
python3.6 sample_generation.py c_deep "$@"


echo "Generating headline popularity samples"
python3.6 sample_generation.py headline "$@"

echo "Generating clickbait samples"
python3.6 sample_generation.py clickbait "$@"

echo "Generating summarizer samples"
python3.6 sample_generation.py summarizers "$@"