"""
This module defines the `Dataset` class and functions for generating datasets for LinHRU/NonLinHRU models.
A `Dataset` object contains raw dataloaders that return raw time series data suitable for 
structured state space models (SSMs).
"""

import os
import pickle
from dataclasses import dataclass
from typing import Dict

import jax.numpy as jnp
import jax.random as jr

from data_dir.dataloaders import Dataloader


@dataclass
class Dataset:
    name: str
    raw_dataloaders: Dict[str, Dataloader]
    data_dim: int
    label_dim: int
    seq_length: int


def dataset_generator(name, data, labels, inmemory=True, *, key):
    """Generate dataset with train/val/test split for LinHRU/NonLinHRU models."""
    if name == "ppg":
        train_data, val_data, test_data = data
        train_labels, val_labels, test_labels = labels
    else:
        N = len(data)
        permkey, key = jr.split(key)
        bound1 = int(N * 0.7)
        bound2 = int(N * 0.85)
        idxs_new = jr.permutation(permkey, N)
        
        train_data, train_labels = (
            data[idxs_new[:bound1]],
            labels[idxs_new[:bound1]],
        )
        val_data, val_labels = (
            data[idxs_new[bound1:bound2]],
            labels[idxs_new[bound1:bound2]],
        )
        test_data, test_labels = data[idxs_new[bound2:]], labels[idxs_new[bound2:]]

    data_dim = train_data.shape[-1]
    seq_length = train_data.shape[1]  # Sequence length
    if len(train_labels.shape) == 1:
        label_dim = 1
    else:
        label_dim = train_labels.shape[-1]

    raw_dataloaders = {
        "train": Dataloader(train_data, train_labels, inmemory),
        "val": Dataloader(val_data, val_labels, inmemory),
        "test": Dataloader(test_data, test_labels, inmemory),
    }
    
    return Dataset(
        name,
        raw_dataloaders,
        data_dim,
        label_dim,
        seq_length,
    )


def create_uea_dataset(data_dir, name, include_time, T, *, key):
    """Load UEA dataset for LinHRU/NonLinHRU models."""
    with open(os.path.join(data_dir, "processed", "UEA", name, "data.pkl"), "rb") as f:
        data = pickle.load(f)
    with open(os.path.join(data_dir, "processed", "UEA", name, "labels.pkl"), "rb") as f:
        labels = pickle.load(f)
    
    onehot_labels = jnp.zeros((len(labels), len(jnp.unique(labels))))
    onehot_labels = onehot_labels.at[jnp.arange(len(labels)), labels].set(1)

    if include_time:
        ts = (T / data.shape[1]) * jnp.repeat(jnp.arange(data.shape[1])[None, :], data.shape[0], axis=0)
        data = jnp.concatenate([ts[:, :, None], data], axis=2)

    return dataset_generator(name, data, onehot_labels, inmemory=True, key=key)


def create_ppg_dataset(data_dir, include_time, T, *, key):
    """Load PPG dataset for LinHRU/NonLinHRU models."""
    with open(os.path.join(data_dir, "processed", "PPG", "ppg", "X_train.pkl"), "rb") as f:
        train_data = pickle.load(f)
    with open(os.path.join(data_dir, "processed", "PPG", "ppg", "y_train.pkl"), "rb") as f:
        train_labels = pickle.load(f)
    with open(os.path.join(data_dir, "processed", "PPG", "ppg", "X_val.pkl"), "rb") as f:
        val_data = pickle.load(f)
    with open(os.path.join(data_dir, "processed", "PPG", "ppg", "y_val.pkl"), "rb") as f:
        val_labels = pickle.load(f)
    with open(os.path.join(data_dir, "processed", "PPG", "ppg", "X_test.pkl"), "rb") as f:
        test_data = pickle.load(f)
    with open(os.path.join(data_dir, "processed", "PPG", "ppg", "y_test.pkl"), "rb") as f:
        test_labels = pickle.load(f)
    
    # Add a dimension to the labels
    train_labels = train_labels[:, :, None]
    val_labels = val_labels[:, :, None]
    test_labels = test_labels[:, :, None]

    if include_time:
        ts = (T / train_data.shape[1]) * jnp.repeat(jnp.arange(train_data.shape[1])[None, :], train_data.shape[0], axis=0)
        train_data = jnp.concatenate([ts[:, :, None], train_data], axis=2)
        ts = (T / val_data.shape[1]) * jnp.repeat(jnp.arange(val_data.shape[1])[None, :], val_data.shape[0], axis=0)
        val_data = jnp.concatenate([ts[:, :, None], val_data], axis=2)
        ts = (T / test_data.shape[1]) * jnp.repeat(jnp.arange(test_data.shape[1])[None, :], test_data.shape[0], axis=0)
        test_data = jnp.concatenate([ts[:, :, None], test_data], axis=2)

    data = (train_data, val_data, test_data)
    labels = (train_labels, val_labels, test_labels)

    return dataset_generator("ppg", data, labels, inmemory=False, key=key)


def create_dataset(data_dir, name, include_time, T, *, key):
    """
    Create dataset for LinHRU/NonLinHRU models.
    Always performs random train/val/test split (no presplit support).
    """
    uea_subfolders = [f.name for f in os.scandir(os.path.join(data_dir, "processed", "UEA")) if f.is_dir()]

    if name in uea_subfolders:
        return create_uea_dataset(data_dir, name, include_time, T, key=key)
    elif name == "ppg":
        return create_ppg_dataset(data_dir, include_time, T, key=key)
    else:
        raise ValueError(f"Dataset {name} not found")
