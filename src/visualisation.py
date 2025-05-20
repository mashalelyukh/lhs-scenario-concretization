import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter


# plots parameter ranges with min/max labels and concrete sample values
def plot_parameter_ranges(numerical_parameters, enum_parameters, concrete_samples):
    # prepare a dictionary to collect all sample values for each parameter name
    collected = defaultdict(list)

    # extract all matching values from the concrete_samples list
    for sample in concrete_samples:
        for full_key, val in sample.items():
            # TODO fix the zeros
            param = full_key.split('_')[0]
            collected[param].append(val)

    num_keys = list(numerical_parameters.keys())
    enum_keys = list(enum_parameters.keys())
    all_keys = num_keys + enum_keys

    total = len(all_keys)
    if total == 0:
        return

    fig, axes = plt.subplots(total, 1,
                             figsize=(10, total * 2.5), sharex=False)
    if total == 1:
        axes = [axes]

    for ax, key in zip(axes, all_keys):
        if key in numerical_parameters:
            info = numerical_parameters[key][0]
            name = info['param']
            lo, hi = info['min'], info['max']
            unit = info.get('unit', '')

            # draw grey range line
            ax.plot([lo, hi], [0, 0], marker='o', color='grey', linewidth=3, zorder=1)

            # count samples
            samples = collected.get(name, [])
            cnt = Counter(samples)
            for val, c in cnt.items():
                label = f"{val:.2f}" if c == 1 else f"{val:.2f}({c})"
                ax.scatter(val, 0, color='black', s=80, zorder=3)
                ax.text(val, 0.05, label, ha='center', va='bottom', fontsize=8, rotation=45, zorder=3)

            # annotate endpoints
            ax.text(lo, -0.15, f"{lo:.2f}", ha='center', va='top', fontsize=10, fontweight='bold')
            ax.text(hi, -0.15, f"{hi:.2f}", ha='center', va='top', fontsize=10, fontweight='bold')

            title = name + (f" [{unit}]" if unit else "")
            ax.set_title(title, loc='left', fontsize=11)
            ax.axis('off')

        else:
            # enum parameter
            vals = enum_parameters[key][0]['values']
            positions = list(range(len(vals)))

            # draw grey range from first to last index
            ax.plot([positions[0], positions[-1]], [0, 0], marker='o', color='grey', linewidth=3, zorder=1)

            # count enum samples
            samples = collected.get(key, [])
            # map enum values to indices
            idxs = [vals.index(v) for v in samples]
            cnt = Counter(idxs)
            for idx, c in cnt.items():
                label = vals[idx] if c == 1 else f"{vals[idx]}({c})"
                ax.scatter(idx, 0, color='black', s=80, zorder=3)
                ax.text(idx, 0.05, label, ha='center', va='bottom', fontsize=8, rotation=45, zorder=3)

            # annotate each tick underline
            for i, v in enumerate(vals):
                ax.text(i, -0.15, f"{v}={i}", ha='center', va='top', fontsize=10, fontweight='bold')

            ax.set_title(key, loc='left', fontsize=11)
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_function_response(f, ast_dict, bo=None, concrete_samples=None, x_sel=None, y_sel=None, resolution=500):
    param_names = list(ast_dict.keys())
    if len(param_names) != 1:
        print("Only 1D plotting is supported.")
        return
    # --- unpack the single parameter's range ----
    p = param_names[0]
    lo = ast_dict[p][0]['min']
    hi = ast_dict[p][0]['max']

    # --- 1D grid of scalars ---
    x_grid = np.linspace(lo, hi, resolution)  # shape (resolution,)

    # --- true objective f(x) in gray ---
    y_true = np.array([f([x]) for x in x_grid])
    # two stacked planes
    fig, (ax_f, ax_u) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8, 6),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )
    ax_f.plot(x_grid, y_true,
              color='gray', lw=2,
              label=fr"objective $f({p})$")

    # --- GP posterior mean & ±1σ if bo is given ---
    if bo is not None:
        # 1) turn x_grid into a list-of-lists [[x], …]
        Xnum = bo.encode(x_grid.reshape(-1, 1).tolist())  # shape (resolution,1)

        # 2) GP predict
        mu, sigma = bo.gp.predict(Xnum, return_std=True)

        # 3) posterior mean
        ax_f.plot(x_grid, mu,
                  color='C0', lw=2,
                  label=r"posterior mean $\mu(x)$")

        # 4) ±σ band
        ax_f.fill_between(x_grid,
                          mu - sigma,
                          mu + sigma,
                          color='C0', alpha=0.2,
                          label=r"uncertainty $\pm\sigma(x)$")

    # --- concrete samples as black dots ---
    if concrete_samples:
        xs = [s[f"{p}_0"] for s in concrete_samples]
        ys = [s['criticality'] for s in concrete_samples]
        ax_f.scatter(xs, ys,
                     color='k', s=30,
                     label="Concrete Samples")

    # ???
    ax_f.set_ylabel("criticality")
    ax_f.grid(True)

    # --- acquisition on twin axis ---
    if bo is not None:
        acq_name = bo.acq_func
        # ax_u = ax_f.twinx()

        # compute acquisition
        ucb = bo.acquisition(mu, sigma)
        ax_u.plot(x_grid, ucb,
                  color='limegreen', lw=2, alpha=0.6,
                  label=fr"$f({acq_name})$$(x)$")

        # mark the K selected points on UCB
        if x_sel is not None:
            sel_x = [x[0] for x in x_sel]
            # encode & predict at just those K points
            Xnum_sel = bo.encode([[xx] for xx in sel_x])
            mu_sel, sigma_sel = bo.gp.predict(Xnum_sel, return_std=True)
            ucb_sel = bo.acquisition(mu_sel, sigma_sel)

            ax_u.scatter(sel_x, ucb_sel,
                         marker='v', color='C3', s=100,
                         label=fr"Selected $f({acq_name})$ pts")
        ax_u.set_ylabel("acquisition")
        ax_u.grid(True)

    ax_u.set_xlabel(p)

    # combine legends from both axes
    h1, l1 = ax_f.get_legend_handles_labels()
    h2, l2 = ax_u.get_legend_handles_labels()
    handles, labels = h1 + h2, l1 + l2
    # else:
    #   handles, labels = ax_f.get_legend_handles_labels()

    # --- styling & legend below x-axis ---
    # ax_f.set_xlabel(p)
    # ax_f.set_ylabel("criticality")
    # ax_f.grid(True)
    # ax_f.set_title("Function & GP posterior + acquisition")

    ax_u.legend(handles, labels,
                loc='upper left',
                bbox_to_anchor=(0, -0.3),
                ncol=3, frameon=False)

    plt.tight_layout()
    plt.show()
