"""
Minimal script to regenerate the gradient comparison plot.
This script runs gradient comparisons for LinHRU and NonlinHRU models,
then generates and saves the comparison plot.
"""

import jax
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import matplotlib.ticker as ticker
from datetime import datetime
from pathlib import Path
from run_experiment import run_experiments


def cosine_similarity(grad1, grad2):
    """Calculate cosine similarity between two gradient vectors"""
    norm1 = jnp.linalg.norm(grad1)
    norm2 = jnp.linalg.norm(grad2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return jnp.sum(grad1 * grad2) / (norm1 * norm2)


def norm_ratio(grad1, grad2):
    """Calculate ratio of norms between two gradient vectors"""
    norm1 = jnp.linalg.norm(grad1)
    norm2 = jnp.linalg.norm(grad2)
    if norm2 == 0:
        return float('inf')
    return norm1 / norm2


def plot_parameter_metrics_bar_scientific_side_by_side(
    grads_rhel_linhru, grads_bptt_linhru, 
    grads_rhel_nonlinhru, grads_bptt_nonlinhru,
    title1="LinHRU", title2="NonlinHRU",
    complex_ssm=False, train_steps=False
):
    """
    Create scientific-styled bar plot comparing metrics (cosine similarity and norm ratio) 
    between Hebbian (RHEL) and BPTT approaches for both LinHRU and NonlinHRU models.
    Plots are arranged horizontally (side by side).
    """
    # Set matplotlib styling with built-in mathtext (no LaTeX installation required)
    plt.rcParams.update({
        "text.usetex": False,  # Use matplotlib's built-in mathtext instead of LaTeX
        "font.family": "serif",  # Serif font looks more professional
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "dejavuserif",  # Use DejaVu for math (looks like Computer Modern)
        "axes.labelsize": 35,
        "font.size": 35,
        "xtick.labelsize": 35,
        "ytick.labelsize": 35
    })
    
    # Set up figure with 2x2 grid (LinHRU and NonlinHRU side by side, Cosine and Norm ratio stacked)
    fig, axs = plt.subplots(2, 2, figsize=(20, 8))
    
    # Add column titles
    axs[0, 0].set_title("linear HSSM", fontsize=40)
    axs[0, 1].set_title("nonlinear HSSM", fontsize=40)
    
    # Calculate metrics and plot for each model type
    for col, (model_type, grads_rhel, grads_bptt) in enumerate([
        (title1, grads_rhel_linhru, grads_bptt_linhru),
        (title2, grads_rhel_nonlinhru, grads_bptt_nonlinhru)
    ]):
        # Determine number of layers
        nbr_layer = len(grads_rhel.blocks)
        
        # Define parameters based on model type
        if model_type == "NonlinHRU":
            params = ["W_diag", "B", "b", "c", "alpha"]
        else:  # LinHRU
            params = ["A_diag", "B"]
            if train_steps:
                params.append("steps")
        
        # Dictionaries to store metrics for each parameter and layer
        cosine_sim = {param: [] for param in params}
        norm_ratio_values = {param: [] for param in params}
        
        # Calculate metrics for each parameter across all layers
        for param in params:
            for i in range(nbr_layer):
                # Extract parameter gradients based on model type
                if model_type == "NonlinHRU":
                    rhel_param = getattr(grads_rhel.blocks[i].ssm, param)
                    bptt_param = getattr(grads_bptt.blocks[i].ssm, param)
                    
                    if len(rhel_param.shape) > 1:
                        rhel_param = rhel_param.flatten()
                        bptt_param = bptt_param.flatten()
                else:  # LinHRU
                    if param == "A_diag":
                        rhel_param = grads_rhel.blocks[i].ssm.A_diag
                        bptt_param = grads_bptt.blocks[i].ssm.A_diag
                    elif param == "B":
                        if complex_ssm:
                            rhel_param = grads_rhel.blocks[i].ssm.B[:,:,:].flatten()
                            bptt_param = grads_bptt.blocks[i].ssm.B[:,:,:].flatten()
                        else:
                            rhel_param = grads_rhel.blocks[i].ssm.B[:,:,0].flatten()
                            bptt_param = grads_bptt.blocks[i].ssm.B[:,:,0].flatten()
                    elif param == "steps" and train_steps:
                        rhel_param = grads_rhel.blocks[i].ssm.steps
                        bptt_param = grads_bptt.blocks[i].ssm.steps
                
                # Calculate metrics
                cosine_sim[param].append(cosine_similarity(rhel_param, bptt_param))
                norm_ratio_values[param].append(norm_ratio(rhel_param, bptt_param))
        
        # Set bar width based on number of layers
        bar_width = 0.8 / nbr_layer
        
        # Scientific color palette - colorblind-friendly
        colors = plt.cm.viridis(np.linspace(0, 1, nbr_layer))
        
        # Plot cosine similarity (top row)
        for p_idx, param in enumerate(params):
            for l_idx in range(nbr_layer):
                # Calculate position for each bar
                pos = p_idx + (l_idx - nbr_layer/2 + 0.5) * bar_width
                # Plot bar
                axs[0, col].bar(pos, cosine_sim[param][l_idx], width=bar_width, 
                         color=colors[l_idx], alpha=0.9, 
                         label=f'Layer {l_idx}' if p_idx == 0 and col == 0 else "", 
                         edgecolor='black', linewidth=0.7)
        
        # Remove x-axis ticks for top row
        axs[0, col].set_xticks([])
        axs[0, col].set_xticklabels([])
        
        # Set titles and labels
        if col == 0:  # Only add y-axis labels to the left plots
            axs[0, col].set_ylabel(r'Cosine Similarity', fontsize=30)
        else:
            axs[0, col].set_ylabel('')  # Remove y-axis label for right plots
            
        axs[0, col].grid(True, linestyle='--', alpha=0.7, zorder=0)
        axs[0, col].set_axisbelow(True)  # Place grid behind bars
        # Apply scientific notation formatting to y-axis
        axs[0, col].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        # Plot norm ratio (bottom row)
        for p_idx, param in enumerate(params):
            for l_idx in range(nbr_layer):
                # Calculate position for each bar
                pos = p_idx + (l_idx - nbr_layer/2 + 0.5) * bar_width
                # Plot bar
                axs[1, col].bar(pos, norm_ratio_values[param][l_idx], width=bar_width, 
                         color=colors[l_idx], alpha=0.9, 
                         edgecolor='black', linewidth=0.7)
        
        # Set x-axis ticks and labels for bottom row
        axs[1, col].set_xticks(range(len(params)))
        
        # Format parameter labels with LaTeX
        param_labels = []
        for param in params:
            if param in ["A_diag", "W_diag"]:
                param_labels.append(r"$\mathbf{A}$")
            elif param == "B_imag":
                param_labels.append(r"$\mathbf{B}_{imag}$")
            elif param == "alpha":
                param_labels.append(r"$\alpha$")
            elif param in ["steps", "c"]:
                param_labels.append(r"$\mathbf{\delta}$")
            elif len(param) == 1:
                param_labels.append(r"$\mathbf{" + param + "}$")
            else:
                param_labels.append(r"$\mathbf{" + param + "}$")
        
        axs[1, col].set_xticklabels(param_labels, fontsize=35)
        
        if col == 0:  # Only add y-axis labels to the left plots
            axs[1, col].set_ylabel(r'Norm Ratio', fontsize=30)
        else:
            axs[1, col].set_ylabel('')  # Remove y-axis label for right plots
            
        axs[1, col].grid(True, linestyle='--', alpha=0.7, zorder=0)
        axs[1, col].set_axisbelow(True)  # Place grid behind bars

        # Apply scientific notation formatting to y-axis
        axs[1, col].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        # Set specified y-ticks for both plots
        axs[0, col].set_yticks([0.5, 1.0])
        axs[1, col].set_yticks([0.5, 1.0])

        # set y-limits for both plots
        axs[0, col].set_ylim(0.3, 1.1)
        
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    
    return fig


def main():
    """Main function to run gradient experiments and generate plot."""
    
    # Configure JAX for double precision
    jax.config.update("jax_enable_x64", True)
    print("Using double precision")
    
    # Common arguments for both models
    parsed_arguments = {
        "dataset_name": ["SelfRegulationSCP1"],
        "seeds": [1234],
        "double_precision": True,
        "complex_ssm": True,
        "train_steps": True,
        "grad_scaler": 1e2,
        "print_steps": 100,
        "scale_grad_only": False,
    }
    experiment_folder = "experiment_configs/repeats"
    
    # Generate unique output directory name
    parsed_arguments["output_parent_dir"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    print("=" * 60)
    print("Running LinHRU experiments...")
    print("=" * 60)
    
    # Run LinHRU experiments
    parsed_arguments["learning_algorithm"] = "BPTT"
    l_bptt_grads_dir = run_experiments(["LinHRU"], parsed_arguments, experiment_folder, test_gradient=True)
    
    parsed_arguments["learning_algorithm"] = "RHEL"
    l_rhel_grads_dir = run_experiments(["LinHRU"], parsed_arguments, experiment_folder, test_gradient=True)
    
    grads_rhel = l_rhel_grads_dir[0]['grads']
    grads_bptt = l_bptt_grads_dir[0]['grads']
    
    print("\n" + "=" * 60)
    print("Running NonlinHRU experiments...")
    print("=" * 60)
    
    # Run NonlinHRU experiments
    parsed_arguments["learning_algorithm"] = "BPTT"
    l_uni_bptt_grads_dir = run_experiments(["NonlinHRU"], parsed_arguments, experiment_folder, test_gradient=True)
    
    parsed_arguments["learning_algorithm"] = "RHEL"
    l_uni_rhel_grads_dir = run_experiments(["NonlinHRU"], parsed_arguments, experiment_folder, test_gradient=True)
    
    grads_uni_rhel = l_uni_rhel_grads_dir[0]['grads']
    grads_uni_bptt = l_uni_bptt_grads_dir[0]['grads']
    
    print("\n" + "=" * 60)
    print("Generating plot...")
    print("=" * 60)
    
    # Generate the plot
    fig = plot_parameter_metrics_bar_scientific_side_by_side(
        grads_rhel,
        grads_bptt,
        grads_uni_rhel,
        grads_uni_bptt,
        title1="LinHRU",
        title2="NonlinHRU",
        complex_ssm=True,
        train_steps=True,
    )
    
    # Create output directory if it doesn't exist
    output_dir = Path("outputs/gradient_comparisons")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the plot
    output_path = output_dir / "static_gradient_comparison.pdf"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\nPlot saved to: {output_path}")
    
    # Also save as PNG for easier viewing
    output_path_png = output_dir / "static_gradient_comparison.png"
    plt.savefig(output_path_png, bbox_inches='tight', dpi=300)
    print(f"Plot also saved as: {output_path_png}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
