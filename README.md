# Seq Boosting

This repository contains the code submitted with our paper for sequence binary classification.
The main model used in the paper is XGBoost, with CatBoost and LightGBM as baselines for comparison.

## Run

```bash
python model/seq_boosting_compare.py
```

## Reproduce Conda Environment (One Line)

```bash
conda create -n boosting_repro python=3.14 -y && conda run -n boosting_repro pip install .
```

## Note on Results

Due to training randomness, outputs under `results/boosting_outputs` are not exactly identical in every run.
However, the relative model performance is consistent with the paper, and the reported models remain stably among the top performers.
