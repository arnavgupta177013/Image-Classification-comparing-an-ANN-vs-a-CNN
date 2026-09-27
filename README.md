# Image Classification: ANN vs CNN

Comparison of a baseline Artificial Neural Network (MLP) and a Convolutional
Neural Network on the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
(6 classes: buildings, forest, glacier, mountain, sea, street).

## Contents
- `Assignment_1_Image_Classification_ANN_vs_CNN.ipynb` — full notebook: data
  loading, preprocessing, both model architectures, training, comparison
  analysis, discussion, and a bonus section on CUDA/TensorRT inference
  optimization.
- `requirements.txt` — Python dependencies.

## Setup

```bash
pip install -r requirements.txt
```

Get the dataset either via KaggleHub (the notebook does this automatically):

```bash
pip install kagglehub
```

or manually from Kaggle, unzipped so you have `seg_train/seg_train/<class>/`
and `seg_test/seg_test/<class>/` folders, with the path set in the notebook's
`DATA_DIR` variable.

## Run

```bash
jupyter notebook Assignment_1_Image_Classification_ANN_vs_CNN.ipynb
```

A CUDA GPU is recommended for reasonable training time but is not required —
the notebook auto-detects and falls back to CPU.

## Summary of findings

The notebook trains both models under matched conditions and reports:
- Trainable parameter counts
- Total training time
- Validation and test accuracy
- An inference-speed benchmark, plus a discussion of how CUDA-level
  (cuDNN autotuning, mixed precision) and TensorRT-level (layer fusion,
  INT8/FP16 calibration) optimizations affect deployed inference latency.

See the notebook's final discussion cells for the full write-up of why the
CNN outperforms the ANN on this spatial image classification task.
