# Learning long range dependencies through time reversal symmetry breaking (NeurIPS 2025, oral)

This  repository contains the official implementation for the paper [Learning long range dependencies through time reversal symmetry breaking](https://arxiv.org/abs/2506.05259) by [Guillaume Pourcel](https://guillaumepourcel.github.io/) and [Maxence Ernoult](https://scholar.google.com/citations?user=wGB7cpUAAAAJ&hl=fr).

This repository is an extension of [https://github.com/tk-rusch/linoss](https://github.com/tk-rusch/linoss). 

--------------------

We propose Recurrent Hamiltonian Echo Learning (RHEL), a forward-only proxy of BPTT, which applies to dissipative-free Hamiltonian systems. It leverages the time-reversal symmetry of Hamiltonian systems, and encodes gradients through trajectory differences when this symmetry is "broken" by error signals. We evaluate RHEL on bespoke SSMs and show it remains on par with BPTT on long-range tasks.

RHEL thus points toward alternative compute paradigms where learning and inference are realized through the same underlying physical dynamics.

<div align="center">
  <img src="media/rhel.png" width="600">
</div>

> Goal: test RHEL at scale by applying it to a SOTA SSM architecture (LinOSS) and a nonlinear SSM (inspired by UnICORNN) to show its generality. RHEL is forward-only: the same forward pass (including parallel scan) is re-used for the backward pass of the Hamiltonian part.
## Requirements

This repository is implemented in python 3.10 and uses Jax as their machine learning framework.

### Environment

This project requires Python 3.10+ and uses JAX for machine learning operations. We recommend using [uv](https://github.com/astral-sh/uv) with the lock file for exact reproducibility:

```bash
pip install uv
uv venv && source .venv/bin/activate
uv pip sync uv.lock
```

Alternatively, you can use `requirements.txt` with pip or uv:

```bash
uv pip install -r requirements.txt
# or: pip install -r requirements.txt
```

#### Verify Installation

After setting up your environment, verify everything is working:

```bash
# Check Python version
python --version  # Should be 3.10+

# Test core imports
python -c "import jax, equinox, optax; print('✓ Environment ready!')"

# Check JAX devices (CPU/GPU)
python -c "import jax; print(f'JAX devices: {jax.devices()}')"
```

#### GPU Support (CUDA)

The default installation uses JAX with CPU-only support. For GPU acceleration, follow the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

---

## Data

The folder `data_dir` contains the scripts for downloading data, preprocessing the data, and creating dataloaders and 
datasets. Raw data should be downloaded into the `data_dir/raw` folder. Processed data should be saved into the `data_dir/processed`
folder in the following format: 
```
processed/{collection}/{dataset_name}/data.pkl, 
processed/{collection}/{dataset_name}/labels.pkl,
processed/{collection}/{dataset_name}/original_idxs.pkl (if the dataset has original data splits)
```
where data.pkl and labels.pkl are jnp.arrays with shape (n_samples, n_timesteps, n_features) 
and (n_samples, n_classes) respectively. If the dataset had original_idxs then those should
be saved as a list of jnp.arrays with shape [(n_train,), (n_val,), (n_test,)].

### The UEA Datasets

The UEA datasets are a collection of multivariate time series classification benchmarks. They can be downloaded by 
running `data_dir/download_uea.py` and preprocessed by running `data_dir/process_uea.py`.

### The PPG-DaLiA Dataset

The PPG-DaLiA dataset is a multivariate time series regression dataset,
where the aim is to predict a person’s heart rate using data
collected from a wrist-worn device. The dataset can be downloaded from the 
<a href="https://archive.ics.uci.edu/dataset/495/ppg+dalia">UCI Machine Learning Repository</a>. The data should be 
unzipped and saved in the `data_dir/raw` folder in the following format `PPG_FieldStudy/S{i}/S{i}.pkl`. The data can be
preprocessed by running the `process_ppg.py` script.

---

## Experiments

The code for training and evaluating the models is contained in `train.py`. Experiments can be run using the `run_experiment.py` script. 
This script requires you to specify the names of the models you want to train, 
the names of the datasets you want to train on, and a directory which contains configuration files. By default,
it will run the LinHRU experiments. The configuration files should be organised as `config_dir/{model_name}/{dataset_name}.json` and contain the
following fields:
- `seeds`: A list of seeds to use for training.
- `data_dir`: The directory containing the data.
- `output_parent_dir`: The directory to save the output.
- `lr_scheduler`: A function which takes the learning rate and returns the new learning rate.
- `num_steps`: The number of steps to train for.
- `print_steps`: The number of steps between printing the loss.
- `batch_size`: The batch size.
- `metric`: The metric to use for evaluation.
- `classification`: Whether the task is a classification task.
- `lr`: The initial learning rate.
- `time`: Whether to include time as a channel.
- `num_blocks`: The number of model blocks.
- `hidden_dim`: The hidden dimension of the model.
- `ssm_dim`: The SSM dimension.
- `complex_ssm`: Whether to use a complex SSM (only for LinHRU).
- `train_steps`: Whether to train the step sizes (only for LinHRU).

See `experiment_configs/repeats` for examples.

### Quick Start: Running Experiments

To run an experiment, you only need to specify the model and dataset. The script will automatically use the default configuration file (which includes 5 seeds by default):

```bash
# Run LinHRU on Heartbeat (the dataset with the smaller number of steps) dataset with BPTT
python run_experiment.py --model_name LinHRU --dataset_name Heartbeat --learning_algorithm BPTT --seeds 2345 

# Run LinHRU on Heartbeat (the dataset with the smaller number of steps) dataset with RHEL
python run_experiment.py --model_name LinHRU --dataset_name Heartbeat --learning_algorithm RHEL --seeds 2345 
```

**Key parameters:**
- `--model_name`: Choose between `LinHRU` or `NonlinHRU`
- `--dataset_name`: Name of the dataset (e.g., `EigenWorms`, `SelfRegulationSCP1`, `ppg`)
- `--learning_algorithm`: Choose between `BPTT` (backpropagation through time) or `RHEL` (Recurrent Hamiltonian Echo Learning)
- `--seeds`: (Optional) Override the seeds from the config file, e.g., `--seeds 2345 3456 4567`

The script will look for the configuration file at `experiment_configs/repeats/{model_name}/{dataset_name}.json`, which contains all hyperparameters and training settings. By default, experiments run with the seeds specified in the config file (typically 5 different random seeds for robust evaluation).

**Comparing BPTT and RHEL:**
To compare the two learning algorithms on the same dataset and model, simply run both commands and the results will be saved in separate directories based on the learning algorithm used.

---

## Project Structure

> The codebase has four main areas:

```
rhel/
├── models/                  # The two model architectures
│   ├── LinHRU.py            #   Linear Hamiltonian Recurrent Unit
│   ├── NonlinHRU.py         #   Nonlinear Hamiltonian Recurrent Unit
│   └── generate_model.py    #   Factory: name -> model instance
│
├── data_dir/                # Data pipeline
│   ├── datasets.py          #   Dataset creation (UEA, PPG-DaLiA)
│   ├── dataloaders.py       #   Batching, shuffling
│   ├── download_uea.py      #   Download scripts
│   ├── process_uea.py       #   ARFF -> JAX arrays
│   └── process_ppg.py       #   PPG-DaLiA processing
│
├── train_and_test.py        # Training loop, loss, eval, checkpointing
├── run_experiment.py        # CLI entry point, loads JSON configs
├── gradient_comparison_bptt_rhel.py   # Static gradient comparison (Fig. 4)
└── experiment_configs/      # Hyperparameters per model x dataset
    └── repeats/{LinHRU,NonlinHRU}/{Dataset}.json
```

### Media and Documentation

- **`media/`**: Contains images and figures used in the README (e.g., RHEL diagram).

- **`LICENSE`**: MIT License for the project.

## Model Architecture
> Both models share the same macro-architecture -- a standard deep SSM design. There are three levels of class hierarchy. The novelty is inside the SSM layer.

#### Shared HSSM architecture (both models)

> Three nested classes: **Model** (top-level), **Block** (repeated N times), **Layer** (the SSM core).

```
┌─────────────────────────────────────────────────────────────┐
│  LinHRU / NonlinHRU  (top-level model class)                │
│                                                             │
│  Input: x  (L, data_dim)                                    │
│                                                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  linear_encoder: Linear(data_dim -> hidden_dim)        │ │
│  └───────────────────────┬────────────────────────────────┘ │
│                          │  (L, H)                          │
│                          ▼                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  LinHRUBlock / NonlinHRUBlock  (block class)           │ │
│  │                                                        │ │
│  │  ┌──────────┐  ┌───────────────────┐  ┌────┐  ┌───┐    │ |
│  │  │ .norm    ├─>│ .ssm              ├─>│GELU├─>│.glu│─┐ │ │
│  │  │BatchNorm │  │ LinHRULayer /     │  └────┘  │GLU │ │ │ │
│  │  └──────────┘  │ NonlinHRULayer    │          └───┘  │ │ │
│  │                │ (Hamiltonian core)│    .drop        │ │ │
│  │    skip ───────┼───────────────────┼────────────>( + ) │ │
│  │                └───────────────────┘                   │ │
│  └───────────────────────┬────────────────────────────────┘ │
│                          │                                  │
│                          ▼   (repeat x num_blocks)          │
│                         ...                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Block N  (same structure)                             │ │
│  └───────────────────────┬────────────────────────────────┘ │
│                          │                                  │
│                          ▼                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  linear_layer: Linear(hidden_dim -> label_dim)         │ │
│  │  Classification: mean-pool over time -> softmax        │ │
│  │  Regression:     subsample -> tanh                     │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```
#### The two SSM layers
**LinHRULayer** -- linear hamiltonian dynamics, parallel via `associative_scan`, O(L log L)

**NonlinHRULayer** -- nonlinear hamiltonian dynamics, sequential Leapfrog via `lax.scan`, O(L)

Both share: project input -> run integrator -> read out with C -> add D skip

### Where RHEL lives

> RHEL only changes the **backward pass** of the ssm via a custom backward pass. Jax AD will automatically use this custom backward pass to chain rules through the ssm layer.

## Reproducing the Results

The configuration files for all the experiments with fixed hyperparameters can be found in the `experiment_configs` folder and
`run_experiment.py` is currently configured to run the repeat experiments on the UEA datasets.


Tor reproduce the Figure 4 of the paper, run `python gradient_comparison_bptt_rhel.py`. This will generate the gradient plots for LinHRU and NonlinHRU on a sample from the SCP1 dataset.

---

# Citation
If you found our work useful in your research, please cite our paper at:
```bibtex
@misc{pourcel2025learninglongrangedependencies,
      title={Learning long range dependencies through time reversal symmetry breaking}, 
      author={Guillaume Pourcel and Maxence Ernoult},
      year={2025},
      eprint={2506.05259},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.05259}, 
}
```
(Also consider starring the project on GitHub.)
