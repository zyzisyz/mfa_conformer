# MFA-Conformer

This repository contains the training code accompanying the paper "MFA-Conformer: Multi-scale Feature Aggregation Conformer for Automatic Speaker Verification", which is submitted to Interspeech 2022.

## Installation

Once you have created your Python environment (Python 3.8+), you can simply create the project and install its requirements:

```bash
pip3 install requirements.txt
```

## Data Preparation

```bash
# format Voxceleb test trial list
rm -rf data; mkdir data
wget -P data/ https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt
python3 scripts/format_trials.py \
            --voxceleb1_root $voxceleb1_dir \
            --src_trials_path data/veri_test.txt \
            --dst_trials_path data/vox1_test.txt

# make csv for voxceleb1&2 dev audio (train_dir)
python3 scripts/build_datalist.py \
        --extension wav \
        --dataset_dir data/$train_dir \
        --data_list_path data/train.csv
```


