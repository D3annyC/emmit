# Explaining Music Models through Input Transformation (EMMIT)

A framework for explaining music and audio models by applying transformations to input audio files (modifying tempo, pitch, mode, structure, instruments) and observing model behaviour on this modified input. It is crucial to understand which transformations should and should not be relevant to prediction in every particular model.

For more details, please refer to the [paper](https://dspace.ut.ee/server/api/core/bitstreams/e452fc17-d4df-43d1-9907-16eb02651234/content).

## Requirements

``` text
python 3.10 +
```

## Installation

clone the repository and run:

``` bash
pip install .
```

## Configuration setup

To configure the audio transformations, please refer to the [configuration file](./configuration.yml).

- `augmented_audio_save_path`: The path to save the augmented audio outputs.
- `augmented_meta_save_path`: The path to save the augmented metadata.
- `mir_dataset_path`: The path to save the data sources.
- `hpss`: Set up harmonic/percussive source separation.
- `tempo_factor`: Set up logspace time stretch, for the detail of parameter, please check [muda](https://muda.readthedocs.io/en/stable/#).
- `keys`: Set up linear pitch shift, for the detail of parameter, please check [muda](https://muda.readthedocs.io/en/stable/#).
- `drc`: Set up dynamic range compression, for the detail of parameter, please check [muda](https://muda.readthedocs.io/en/stable/#).

## Usage Examples

The usage example, please refer to the [example.ipynb](./example.ipynb). This notebook demonstrates how to use the library to generate augmented data, train a model, and generate explanations. The example model is in [model](./model/). The example data to show the usage of library is in [data](./data/).

## Citation

If you find this repository useful for your research, please consider citing the following paper:

``` text
@article{emmit,
  title={Audio Transformations Based
Explanations (ATBE) for deep learning
models trained on musical data},
  author={ChengHan Chung, Anna Aljanaki},
  journal={},
  year={2024}
}
```
