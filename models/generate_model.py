"""
This module provides a function to generate LinHRU/NonlinHRU models.

Function:
- `create_model`: Generates and returns a LinHRU or NonlinHRU model instance along with its state
  based on the provided model name and hyperparameters.

Main Parameters for LinHRU/NonlinHRU:
- `model_name`: 'LinHRU' or 'NonlinHRU'
- `data_dim`: The input data dimension.
- `seq_length`: The sequence length of the input data.
- `label_dim`: The output label dimension.
- `hidden_dim`: The hidden state dimension for the model.
- `num_blocks`: The number of blocks (layers) in the model.
- `ssm_dim`: The state-space model dimension.
- `ssm_blocks`: The number of SSM blocks.
- `classification`: A boolean indicating whether the task is classification (True) or regression (False).
- `output_step`: The step interval for outputting predictions in sequence models.
- `learning_algorithm`: Learning algorithm method ('BPTT', 'RHEL', etc.)
- `epsilon`: Regularization parameter for RHEL algorithm.
- `complex_ssm`: Whether to use complex SSM parameters.
- `train_steps`: Whether to train step size parameters.
- `key`: A JAX PRNG key for random number generation.

Returns:
- A tuple containing the created model and its state.

Raises:
- `ValueError`: If required hyperparameters for the specified model are not provided or if an
  unknown model name is passed.
"""

import equinox as eqx
import jax.random as jr

from models.LinHRU import LinHRU
from models.NonlinHRU import NonlinHRU


def create_model(
    model_name,
    data_dim,
    seq_length,
    label_dim,
    hidden_dim,
    num_blocks=None,
    classification=True,
    output_step=1,
    ssm_dim=None,
    learning_algorithm='BPTT',
    epsilon=0.,
    complex_ssm=False,
    train_steps=False,
    *,
    key,
):
    if model_name == "LinHRU":
        if num_blocks is None:
            raise ValueError("Must specify num_blocks for LinHRU.")
        if ssm_dim is None:
            raise ValueError("Must specify ssm_dim for LinHRU.")
        ssm = LinHRU(
            num_blocks,
            data_dim,
            ssm_dim,
            hidden_dim,
            label_dim,
            classification,
            output_step,
            learning_algorithm,
            epsilon,
            complex_ssm,
            train_steps,
            seq_length,
            key=key,
        )
        state = eqx.nn.State(ssm)
        return ssm, state
    elif model_name == "NonlinHRU":
        if num_blocks is None:
            raise ValueError("Must specify num_blocks for NonlinHRU.")
        if ssm_dim is None:
            raise ValueError("Must specify ssm_dim for NonlinHRU.")
        ssm = NonlinHRU(
            num_blocks,
            data_dim,
            ssm_dim,
            hidden_dim,
            label_dim,
            classification,
            output_step,
            learning_algorithm,
            epsilon,
            seq_length,
            key=key,
        )
        state = eqx.nn.State(ssm)
        return ssm, state
    else:
        raise ValueError(f"Unknown model name: {model_name}. Only 'LinHRU' and 'NonlinHRU' are supported.")
