# Automatic Pull Quote Selection
_Learning_ to automatically select good pull quotes.

:construction: This code accompanies the paper [Catching Attention with Automatic Pull Quote Selection](https://arxiv.org/abs/2005.13263).

## Preparing the dataset
To reproduce our dataset:
1. navigate to the `datasets/url_lists/` directory and unzip `url_lists.zip` so that the 4 files are in `datasets/url_lists/`
2. nagivate to `datasets/` and run `python3.6 construct_dataset.py source my_save_dir/`.
   * source can be one of `intercept`, `ottawa-citizen`, `cosmo`, `national-post`, or `all`
   * the samples for a given source will be stored in `my_save_dir/source/`
   * :warning: Update `settings.py` so that `base_pq_directory` points to `my_save_dir/`.
   * :warning: This will take a long time. To avoid potential legal difficulties, we avoid providing the fully prepared dataset here. Email tannerbohn@gmail.com for more info.
3. navigate to the root repo folder and run `python3.6 calculate_data_stats.py` to calculate dataset statistics to compare with our paper.

## Reproducing experiments
To reproduce out experimental results (stored in `/results`), run `bash run_experiments.sh`

:information_source: To first make sure that things work, run `bash run_experiments.sh --quick`. It should take just a few minutes.

## Miscellaneous

To reproduce the handcrafted feature value distribution figures, run `python3.6 view_feature_dists.py`

To analyze test articles with a specific model, run `bash generate_model_samples.sh`. The `--quick` argument can similarly be used to make sure things are working.
