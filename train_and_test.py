"""Training and evaluation utilities for LinHRU and NonlinHRU models.



This module provides a complete pipeline for training and testing recurrent state-space models
(LinHRU/NonlinHRU) on time series classification and regression tasks using JAX and Equinox.

Main Components:
    - Dataset creation and loading
    - Model initialization with various learning algorithms (BPTT, RTRL, RHEL)
    - Training loop with automatic checkpointing and validation
    - Gradient computation and analysis utilities

Key Functions:
    calc_output: Computes model predictions with support for stateful and non-deterministic models.
        Automatically handles batch processing using JAX's vmap.
    
    classification_loss: Cross-entropy loss for classification tasks.
    
    regression_loss: Mean squared error loss for regression tasks.
    
    make_step: Executes a single optimization step using Adam optimizer with gradient scaling support.
    
    train_model: Main training loop with early stopping, periodic evaluation, and automatic 
        model checkpointing. Supports both accuracy and MSE metrics.
    
    test_gradient_model: Computes and optionally saves gradients for a single batch, useful for
        gradient analysis and debugging.
    
    create_dataset_model_and_train: End-to-end pipeline that creates dataset, initializes model,
        and runs training. Supports various learning algorithms and configuration options.

"""

import os
import shutil
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jax.tree_util import Partial


from data_dir.datasets import create_dataset
from models.generate_model import create_model


def metric_regression(prediction, y):
    return jnp.mean(jnp.mean(jnp.mean((prediction - y) ** 2, axis=2), axis=1), axis=0)


@eqx.filter_jit
def calc_output(model, X, state, key, stateful, nondeterministic):
    if stateful:
        if nondeterministic:
            output, state = jax.vmap(model, axis_name="batch", in_axes=(0, None, None), out_axes=(0, None))(X, state, key)
        else:
            output, state = jax.vmap(model, axis_name="batch", in_axes=(0, None), out_axes=(0, None))(X, state)
    elif nondeterministic:
        output = jax.vmap(model, in_axes=(0, None))(X, key)
    else:
        output = jax.vmap(model)(X)

    return output, state


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def classification_loss(diff_model, static_model, X, y, state, key, grad_scaler):
    model = eqx.combine(diff_model, static_model)
    pred_y, state = calc_output(model, X, state, key, model.stateful, model.nondeterministic)
    return (
        jnp.mean(-jnp.sum(y * jnp.log(pred_y + 1e-8), axis=1)) * grad_scaler,
        state,
    )


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def regression_loss(diff_model, static_model, X, y, state, key, grad_scaler):
    model = eqx.combine(diff_model, static_model)
    pred_y, state = calc_output(model, X, state, key, model.stateful, model.nondeterministic)
    return (
        jnp.mean(jnp.mean(jnp.mean((pred_y - y) ** 2, axis=2), axis=1)) * grad_scaler,
        state,
    )




@eqx.filter_jit
def make_step(model, filter_spec, X, y, loss_fn, state, opt, opt_state, key, grad_scaler, scale_grad_only):
    diff_model, static_model = eqx.partition(model, filter_spec)
    (value, state), grads = loss_fn(diff_model, static_model, X, y, state, key, grad_scaler)
    # rescale gradients
    if scale_grad_only:
        grads = jax.tree_util.tree_map(lambda x: x * grad_scaler, grads)
    else:
        grads = jax.tree_util.tree_map(lambda x: x / grad_scaler, grads)
    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, state, opt_state, value


@eqx.filter_jit
def get_grad(model, filter_spec, X, y, loss_fn, state, opt, opt_state, key, grad_scaler):
    diff_model, static_model = eqx.partition(model, filter_spec)
    (value, state), grads = loss_fn(diff_model, static_model, X, y, state, key, grad_scaler)
    # rescale gradients
    grads = jax.tree_util.tree_map(lambda x: x / grad_scaler, grads)
    return model, state, opt_state, value, grads


def save_model_and_gradients(model, grads, output_dir):
    """Save model and gradients to disk with error handling."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "model.eqx")
        grads_path = os.path.join(output_dir, "gradients.eqx")

        eqx.tree_serialise_leaves(model_path, model)
        eqx.tree_serialise_leaves(grads_path, grads)

        print(f"Model saved to: {model_path}")
        print(f"Gradients saved to: {grads_path}")
        return True
    except Exception as e:
        print(f"Error saving model and gradients: {e}")
        return False


def train_model(
    dataset_name,
    model,
    metric,
    filter_spec,
    state,
    dataloaders,
    num_steps,
    print_steps,
    lr,
    lr_scheduler,
    batch_size,
    key,
    output_dir,
    grad_scaler,
    scale_grad_only=False,
):

    if metric == "accuracy":
        best_val = max
        operator_improv = lambda x, y: x >= y
        operator_no_improv = lambda x, y: x <= y
    elif metric == "mse":
        best_val = min
        operator_improv = lambda x, y: x <= y
        operator_no_improv = lambda x, y: x >= y
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if os.path.isdir(output_dir):
        user_input = input(f"Warning: Output directory {output_dir} already exists. Do you want to delete it? (yes/no): ")
        if user_input.lower() == "yes":
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            print(f"Directory {output_dir} has been deleted and recreated.")
        else:
            raise ValueError(f"Directory {output_dir} already exists. Exiting.")
    else:
        os.makedirs(output_dir)
        print(f"Directory {output_dir} has been created.")

    batchkey, key = jr.split(key, 2)
    opt = optax.adam(learning_rate=lr_scheduler(lr))
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    if model.classification:
        loss_fn = classification_loss
    else:
        loss_fn = regression_loss

    running_loss = 0.0
    if metric == "accuracy":
        all_val_metric = [0.0]
        all_train_metric = [0.0]
        val_metric_for_best_model = [0.0]
    elif metric == "mse":
        all_val_metric = [100.0]
        all_train_metric = [100.0]
        val_metric_for_best_model = [100.0]
    no_val_improvement = 0
    all_time = []
    all_memory = []
    training_start = time.time()
    for step, data in zip(
        range(num_steps),
        dataloaders["train"].loop(batch_size, key=batchkey),
    ):
        stepkey, key = jr.split(key, 2)
        X, y = data
        model, state, opt_state, value = make_step(model, filter_spec, X, y, loss_fn, state, opt, opt_state, stepkey, grad_scaler, scale_grad_only)
        # print(f"Step: {step + 1}, Loss: {value}")
        running_loss += value
        if (step + 1) % print_steps == 0:
            predictions = []
            labels = []
            i = 0
            for data in dataloaders["train"].loop_epoch(batch_size):
                stepkey, key = jr.split(key, 2)
                inference_model = eqx.tree_inference(model, value=True)
                X, y = data
                prediction, _ = calc_output(
                    inference_model,
                    X,
                    state,
                    stepkey,
                    model.stateful,
                    model.nondeterministic,
                )
                predictions.append(prediction)
                labels.append(y)
            prediction = jnp.vstack(predictions)
            y = jnp.vstack(labels)
            
            if model.classification:
                train_metric = jnp.mean(jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1))
                train_loss = jnp.mean(-jnp.sum(y * jnp.log(prediction + 1e-8), axis=1))
            else:
                prediction = prediction[:, :, :]
                train_metric = metric_regression(prediction, y)
                train_loss = train_metric
            predictions = []
            labels = []
            i = 0
            for data in dataloaders["val"].loop_epoch(batch_size):
                stepkey, key = jr.split(key, 2)
                inference_model = eqx.tree_inference(model, value=True)
                X, y = data
                prediction, _ = calc_output(
                    inference_model,
                    X,
                    state,
                    stepkey,
                    model.stateful,
                    model.nondeterministic,
                )
                predictions.append(prediction)
                labels.append(y)

            prediction = jnp.vstack(predictions)
            y = jnp.vstack(labels)
            
            if model.classification:
                val_metric = jnp.mean(jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1))
                val_loss = jnp.mean(-jnp.sum(y * jnp.log(prediction + 1e-8), axis=1))
            else:
                prediction = prediction[:, :, :]
                val_metric = metric_regression(prediction, y)
                val_loss = jnp.mean(jnp.mean(jnp.mean((prediction - y) ** 2, axis=2), axis=1))

            end = time.time()
            total_time = end - training_start
            
            
            avg_loss = running_loss / print_steps
            print(
                f"Step: {step + 1}, Loss: {avg_loss}, "
                f"Train metric: {train_metric}, Train loss: {train_loss}, "
                f"Validation metric: {val_metric}, Val loss: {val_loss}, "
            )
            
            if step > 0:
                if operator_no_improv(val_metric, best_val(val_metric_for_best_model)):
                    no_val_improvement += 1
                    if no_val_improvement > 10:
                        break
                else:
                    no_val_improvement = 0
                if operator_improv(val_metric, best_val(val_metric_for_best_model)):
                    val_metric_for_best_model.append(val_metric)
                    predictions = []
                    labels = []
                    for data in dataloaders["test"].loop_epoch(batch_size):
                        stepkey, key = jr.split(key, 2)
                        inference_model = eqx.tree_inference(model, value=True)
                        X, y = data
                        prediction, _ = calc_output(
                            inference_model,
                            X,
                            state,
                            stepkey,
                            model.stateful,
                            model.nondeterministic,
                        )
                        predictions.append(prediction)
                        labels.append(y)
                    prediction = jnp.vstack(predictions)
                    y = jnp.vstack(labels)
                    if model.classification:
                        test_metric = jnp.mean(jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1))
                    else:
                        prediction = prediction[:, :, :]
                        test_metric = metric_regression(prediction, y)
                    print(f"Test metric: {test_metric}")
                    
                running_loss = 0.0
                all_train_metric.append(train_metric)
                all_val_metric.append(val_metric)
                all_time.append(total_time)
                steps = jnp.arange(0, step + 1, print_steps)
                all_train_metric_save = jnp.array(all_train_metric)
                all_val_metric_save = jnp.array(all_val_metric)
                all_time_save = jnp.array(all_time)
                all_memory_save = jnp.array(all_memory)
                test_metric_save = jnp.array(test_metric)
                jnp.save(os.path.join(output_dir, "steps.npy"), steps)
                jnp.save(os.path.join(output_dir, "all_train_metric.npy"), all_train_metric_save)
                jnp.save(os.path.join(output_dir, "all_val_metric.npy"), all_val_metric_save)
                jnp.save(os.path.join(output_dir, "all_time.npy"), all_time_save)
                jnp.save(os.path.join(output_dir, "all_memory.npy"), all_memory_save)
                jnp.save(os.path.join(output_dir, "test_metric.npy"), test_metric_save)

        # End the run if time goes over 4 hours (compute cluster limit used for experiments)
        end_training = time.time()
        total_time = end_training - training_start
        if total_time > 4 * 3600:
            print("Time limit exceeded 4 hours, ending training.")
            break
    
    print(f"Test metric: {test_metric}")

    steps = jnp.arange(0, num_steps + 1, print_steps)
    all_train_metric = jnp.array(all_train_metric)
    all_val_metric = jnp.array(all_val_metric)
    all_time = jnp.array(all_time)
    test_metric = jnp.array(test_metric)
    jnp.save(os.path.join(output_dir, "steps.npy"), steps)
    jnp.save(os.path.join(output_dir, "all_train_metric.npy"), all_train_metric)
    jnp.save(os.path.join(output_dir, "all_val_metric.npy"), all_val_metric)
    jnp.save(os.path.join(output_dir, "all_time.npy"), all_time)
    jnp.save(os.path.join(output_dir, "test_metric.npy"), test_metric)

    return model


def test_gradient_model(
    model,
    filter_spec,
    state,
    dataloaders,
    lr,
    lr_scheduler,
    batch_size,
    key,
    output_dir,
    grad_scaler,
):
    """Compute gradients for one batch and optionally save them."""
    if os.path.isdir(output_dir):
        user_input = input(f"Warning: Output directory {output_dir} already exists. Do you want to delete it? (yes/no): ")
        if user_input.lower() == "yes":
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            print(f"Directory {output_dir} has been deleted and recreated.")
        else:
            raise ValueError(f"Directory {output_dir} already exists. Exiting.")
    else:
        os.makedirs(output_dir)
        print(f"Directory {output_dir} has been created.")

    batchkey, key = jr.split(key, 2)
    opt = optax.adam(learning_rate=lr_scheduler(lr))
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    if model.classification:
        loss_fn = classification_loss
    else:
        loss_fn = regression_loss

    batch_size = 1
    # Get one batch and compute gradients
    for data in dataloaders["train"].loop(batch_size, key=batchkey):
        stepkey, key = jr.split(key, 2)
        X, y = data
        print("grad_scaler", grad_scaler)
        model, state, opt_state, value, grads = get_grad(
            model, filter_spec, X, y, loss_fn, state, opt, opt_state, stepkey, grad_scaler
        )
        break

    save_model_and_gradients(model, grads, output_dir)
    return grads, output_dir


def create_dataset_model_and_train(
    seed,
    data_dir,
    dataset_name,
    output_step,
    metric,
    include_time,
    model_name,
    learning_algorithm,
    epsilon,
    complex_ssm,
    train_steps,
    model_args,
    num_steps,
    print_steps,
    lr,
    lr_scheduler,
    batch_size,
    grad_scaler=1.0,
    scale_grad_only=False,
    output_parent_dir="",
    test_gradient=False,
):
    """Create dataset, model, and train for LinHRU/NonlinHRU models."""
    model_name_directory = model_name + "_" + learning_algorithm
    
    if test_gradient:
        output_parent_dir = os.path.join("outputs_grad", output_parent_dir, dataset_name, model_name_directory)
    else:
        output_parent_dir = os.path.join("outputs", output_parent_dir, model_name_directory)
    
    output_dir = f"T_1.00_time_{include_time}_nsteps_{num_steps}_lr_{lr}"
    
    if learning_algorithm == "RHEL":
        output_dir += f"_epsilon_{epsilon:.2f}"
    
    for k, v in model_args.items():
        name = str(v)
        if "(" in name:
            name = name.split("(", 1)[0]
        output_dir += f"_{k}_" + name
    
    output_dir += f"_seed_{seed}"
    if test_gradient:
        output_dir = dataset_name + "_" + output_dir

    key = jr.PRNGKey(seed)
    datasetkey, modelkey, trainkey, key = jr.split(key, 4)
    
    print(f"Creating dataset {dataset_name}")
    dataset = create_dataset(
        data_dir,
        dataset_name,
        include_time=include_time,
        T=1.0,
        key=datasetkey,
    )

    print(f"Creating model {model_name}")
    classification = metric == "accuracy"
    model, state = create_model(
        model_name,
        dataset.data_dim,
        dataset.seq_length,  
        dataset.label_dim,
        classification=classification,
        output_step=output_step,
        learning_algorithm=learning_algorithm,
        epsilon=epsilon,
        complex_ssm=complex_ssm,
        train_steps=train_steps,
        **model_args,
        key=modelkey,
    )
    
    filter_spec = jax.tree_util.tree_map(lambda _: True, model)
    dataloaders = dataset.raw_dataloaders

    if test_gradient:
        return test_gradient_model(
            model,
            filter_spec,
            state,
            dataloaders,
            lr,
            lr_scheduler,
            batch_size,
            trainkey,
            os.path.join(output_parent_dir, output_dir),
            grad_scaler,
        )
    
    return train_model(
        dataset_name,
        model,
        metric,
        filter_spec,
        state,
        dataloaders,
        num_steps,
        print_steps,
        lr,
        lr_scheduler,
        batch_size,
        trainkey,
        os.path.join(output_parent_dir, output_dir),
        grad_scaler,
        scale_grad_only=scale_grad_only,
    )
