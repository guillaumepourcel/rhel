"""
This script loads hyperparameters from JSON files and trains LinHRU/NonlinHRU models on specified datasets.
The results are saved in output directories defined in the JSON files.

The `run_experiments` function iterates over model names and dataset names, loading configuration
files from a specified folder, and calls the training function.

Usage:
- Use command-line arguments to specify dataset, model, seeds, and other parameters.
"""

import argparse
import json
import os
from train_and_test import create_dataset_model_and_train
import jax


def run_experiments(model_names, parsed_arguments, experiment_folder, test_gradient=False, model_args=None):
    """Run experiments for LinHRU/NonlinHRU models."""
    dataset_names = parsed_arguments["dataset_name"]
    seeds = parsed_arguments["seeds"]
    learning_algorithm = parsed_arguments["learning_algorithm"]
    
    for model_name in model_names:
        for dataset_name in dataset_names:
            with open(experiment_folder + f"/{model_name}/{dataset_name}.json", "r") as file:
                data = json.load(file)
            print(data)
            print(f"overriding with {parsed_arguments}")
            
            data_dir = data["data_dir"]
            lr_scheduler = eval(data["lr_scheduler"])
            num_steps = data["num_steps"]
            print_steps = parsed_arguments["print_steps"]
            batch_size = data["batch_size"]
            metric = data["metric"]
            
            if learning_algorithm == "RHEL":
                epsilon = data["epsilon"]
            else:
                epsilon = None
            
            lr = float(data["lr"])
            include_time = data["time"].lower() == "true"
            hidden_dim = int(data["hidden_dim"])
            ssm_dim = int(data["ssm_dim"])
            num_blocks = int(data["num_blocks"])
            
            if dataset_name == "ppg":
                output_step = int(data["output_step"])
            else:
                output_step = 1

            model_args_default = {
                "num_blocks": num_blocks,
                "hidden_dim": hidden_dim,
                "ssm_dim": ssm_dim,
            }
            
            # Merge with additional model_args if provided
            if model_args is not None:
                model_args_default.update(model_args)
            model_args_final = model_args_default
            
            run_args = {
                "data_dir": data_dir,
                "dataset_name": dataset_name,
                "output_step": output_step,
                "metric": metric,
                "include_time": include_time,
                "model_name": model_name,
                "learning_algorithm": learning_algorithm,
                "epsilon": epsilon,
                "complex_ssm": parsed_arguments["complex_ssm"],
                "train_steps": parsed_arguments["train_steps"],
                "model_args": model_args_final,
                "num_steps": num_steps,
                "print_steps": print_steps,
                "lr": lr,
                "lr_scheduler": lr_scheduler,
                "batch_size": batch_size,
                "output_parent_dir": parsed_arguments["output_parent_dir"],
                "test_gradient": test_gradient,
                "grad_scaler": parsed_arguments["grad_scaler"],
                "scale_grad_only": parsed_arguments["scale_grad_only"],
            }
            
            run_fn = create_dataset_model_and_train
            l_grads_dir = []
            
            for seed in seeds:
                print(f"Running experiment with seed: {seed}")
                if test_gradient:
                    grads, dir = run_fn(seed=seed, **run_args)
                    l_grads_dir.append({"grads": grads, "dir": dir})
                    return l_grads_dir
                else:
                    run_fn(seed=seed, **run_args)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_name", type=str, default="SelfRegulationSCP1")
    args.add_argument("--seeds", nargs="+", type=int, help="List of integers", default=[2345, 3456, 4567])
    args.add_argument("--double_precision", action=argparse.BooleanOptionalAction, default=False, help="Use double precision")
    args.add_argument("--complex_ssm", action=argparse.BooleanOptionalAction, default=True, help="Use complex SSM")
    args.add_argument("--train_steps", action=argparse.BooleanOptionalAction, default=True, help="Train steps")
    args.add_argument("--learning_algorithm", type=str, default="BPTT", help="Learning algorithm method")
    args.add_argument("--output_parent_dir", type=str, default="", help="change the name of the output directory")
    args.add_argument("--grad_scaler", type=float, default=1.0, help="Gradient scaler for training")
    args.add_argument("--model_name", type=str, default="NonlinHRU", help="Model name to run")
    args.add_argument("--print_steps", type=int, default=1000, help="Print steps")
    args.add_argument("--no_jax_prealloc", action="store_true", default=False, help="Disable JAX memory pre-allocation")
    args.add_argument("--scale_grad_only", action="store_true", default=False, help="Reproduce gradient scaling that was used in the paper (loss is scaled but the gradient is not downscaled before being applied)")
    args = args.parse_args()
    if args.double_precision:
        jax.config.update("jax_enable_x64", True)
    
    # Set JAX memory pre-allocation based on flag
    if args.no_jax_prealloc:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        print("JAX memory pre-allocation disabled")
    else:
        print("JAX memory pre-allocation enabled (default)")

    model_names = [args.model_name]
    print(f"Running {model_names} on {args.dataset_name}")
    parsed_arguments = {
        "dataset_name": [args.dataset_name],
        "seeds": args.seeds,
        "double_precision": args.double_precision,
        "complex_ssm": args.complex_ssm,
        "train_steps": args.train_steps,
        "output_parent_dir": args.output_parent_dir,
        "learning_algorithm": args.learning_algorithm,
        "grad_scaler": args.grad_scaler,
        "print_steps": args.print_steps,
        "scale_grad_only": args.scale_grad_only,
    }
    experiment_folder = "experiment_configs/repeats"

    run_experiments(model_names, parsed_arguments, experiment_folder)
