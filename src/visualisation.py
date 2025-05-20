import re

import matplotlib.pyplot as plt
import numpy as np

def plot_parameter_ranges(numerical_parameters, concrete_samples):
    """
    Plots parameter ranges with min/max labels and concrete sample values.

    Parameters:
    - numerical_parameters: list of dicts, each with 'param', 'min', 'max', 'unit', etc.
    - concrete_samples: list of dicts, each mapping parameter keys to values.
                        Keys may include suffixes like 'speed_0', 'speed1_0' etc.
                        Matching is based on parameter name prefix.
    """
    import re
    from collections import defaultdict

    # Prepare a dictionary to collect all sample values for each parameter name
    collected_samples = defaultdict(list)


    # Extract all matching values from the concrete_samples list
    for sample_dict in concrete_samples:
        for full_key, value in sample_dict.items():
            # TODO fix the zeros
            param = full_key.split('_')[0]
            collected_samples[param].append(value)

    num_vars = len(numerical_parameters)
    fig, axes = plt.subplots(num_vars, 1, figsize=(10, num_vars * 2.5), sharex=False)

    if num_vars == 1:
        axes = [axes]

    for ax, key in zip(axes, numerical_parameters.keys()):
        param = numerical_parameters[key][0]
        name = param['param']
        min_val = param['min']
        max_val = param['max']
        unit = param.get('unit', '')

        samples = collected_samples.get(name, [])

        # Plot the range line with min/max markers
        ax.plot([min_val, max_val], [0, 0], marker='o', linewidth=3, color='grey')

        # Plot and label concrete sample points
        for val in samples:
            ax.scatter(val, 0, color='black')
            ax.text(val, 0.05, f"{val:.2f}", ha='center', va='bottom', fontsize=8, rotation=45)

        # Label min and max under the endpoint dots
        ax.text(min_val, -0.15, f"{min_val:.2f}", ha='center', va='top', fontsize=10, fontweight='bold')
        ax.text(max_val, -0.15, f"{max_val:.2f}", ha='center', va='top', fontsize=10, fontweight='bold')

        # Clean up plot
        ax.axis('off')
        title = f"{name}"
        if unit:
            title += f" [{unit}]"
        ax.set_title(title, fontsize=11, loc='left')

    plt.tight_layout()
    plt.show()


def plot_function_response(f, ast_dict, concrete_samples=None, x_sel=None, y_sel=None, resolution=500):
    """
    Plots the output of a multivariate function f over the ranges defined in ast_dict.
    Only supports 1D or 2D parameter plots (more than 2 params won't be visualized directly).

    Parameters:
    - f: a function that takes a list of floats and returns a float in [0, 1]
    - ast_dict: a dictionary with parameter names as keys and lists of parameter definitions (with 'min' and 'max')
    - resolution: number of samples per dimension
    """
    param_names = list(ast_dict.keys())
    param_defs = [ast_dict[name][0] for name in param_names]
    ranges = [(param['min'], param['max']) for param in param_defs]

    if len(param_names) == 1:
        x_vals = np.linspace(ranges[0][0], ranges[0][1], resolution)
        y_vals = [f([x]) for x in x_vals]

        plt.figure(figsize=(8, 4))
        plt.plot(x_vals, y_vals, label=f"f({param_names[0]})", linewidth=2)

        # Extract labels for axes
        x_label = param_names[0]
        y_label = "f2 output"

        # Plot black concrete sample points
        if concrete_samples:
            first_sample = concrete_samples[0]
            sample_keys = list(first_sample.keys())
            x_key = next(k for k in sample_keys if k != 'criticality')
            y_key = 'criticality'

            sample_x = [s[x_key] for s in concrete_samples]
            sample_y = [s[y_key] for s in concrete_samples]

            x_label = re.sub(r"_\d+$", "", x_key)
            y_label = y_key

            plt.scatter(sample_x, sample_y, color='black', label='Concrete Samples')

        # Plot red selected points
        if x_sel is not None and y_sel is not None:
            x_sel_flat = [x[0] for x in x_sel]
            plt.scatter(x_sel_flat, y_sel, color='red', label='Selected Samples')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title("Function f2 Output Over Parameter Range")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


    elif len(param_names) == 2:
        print("2D plotting not supported with concrete samples in this version.")
    else:
        print("Only 1D parameter plots are supported.")
