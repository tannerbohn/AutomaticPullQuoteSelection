# Automatic Pull Quote Selection
_Learning_ to automatically select good pull quotes.

:construction: This code will accompany the paper [Catching Attention with Automatic Pull Quote Selection](https://arxiv.org/abs/2005.13263).

## Preparing the dataset
To reproduce our dataset, run `python3.6 construct_dataset.py my_save_dir/`. This saves the prepared data in `my_save_dir/`.

:warning: Update `settings.py` so that `PQ_SAMPLES_DIRS` points to the appropriate locations.

:warning: This will take a long time. To avoid potential legal difficulties, we avoid providing the fully prepared dataset here. Email tannerbohn@gmail.com for more info.

To calculate datatset statistics to compare to the table in our paper, run `python3.6 calculate_data_stats.py`.

## Reproducing experiments
To reproduce out experimental results (stored in `/results`), run `bash run_experiments.sh`

:information_source: To first make sure that things work, run `bash run_experiments.sh --quick`. It should take just a few minutes.

## Miscellaneous

To reproduce the handcrafted feature value distribution figures, run `python3.6 view_feature_dists.py`

To analyze test articles with a specific model, run `bash generate_model_samples.sh`. The `--quick` argument can similarly be used to make sure things are working.
