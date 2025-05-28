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


def plot_function_response(
        f,
        numerical_parameters: dict,
        enum_parameters: dict = None,
        bo=None,
        concrete_samples=None,
        x_sel=None,
        resolution=500
):
    enum_parameters = enum_parameters or {}
    # combine keys: must have exactly one total param
    keys = list(numerical_parameters.keys()) + list(enum_parameters.keys())

    # Special case: exactly one numerical parameter AND exactly one enum parameter
    if len(numerical_parameters) == 1 and len(enum_parameters) == 1:
        num_param = list(numerical_parameters.keys())[0]
        enum_param = list(enum_parameters.keys())[0]

        # Get enum values and numerical param range
        enum_vals = enum_parameters[enum_param][0]['values']
        N_enum = len(enum_vals)
        lo = numerical_parameters[num_param][0]['min']
        hi = numerical_parameters[num_param][0]['max']

        # Create a figure with 2 rows (objective and acquisition) and N_enum columns
        fig, axes = plt.subplots(
            2, N_enum,
            figsize=(5 * N_enum, 8),
            sharex='col',
            sharey='row',
            gridspec_kw={'height_ratios': [3, 1]}
        )

        # If only one enum value, ensure axes is still a 2D array
        if N_enum == 1:
            axes = np.array(axes).reshape(2, 1)

        # Prepare x grid for numerical parameter
        x_grid = np.linspace(lo, hi, resolution)

        # Loop through each enum value
        for i, enum_val in enumerate(enum_vals):
            ax_f = axes[0, i]  # Top row for objective function
            ax_u = axes[1, i]  # Bottom row for acquisition function

            # Function to map [num_val, enum_val] to proper encoding for the objective function
            def eval_f_at(x):
                # For the test function, encode enum as i+1
                return f([x, i + 1])

            # Calculate objective function values
            y_obj = np.array([eval_f_at(x) for x in x_grid])

            # Plot objective function
            ax_f.plot(x_grid, y_obj, color='gray', lw=2, label=fr"objective $f({num_param}|{enum_param}={enum_val})$")

            # Add title to the column
            ax_f.set_title(f"{enum_param} = {enum_val}", fontsize=12)

            # GP posterior and acquisition if available
            if bo is not None:
                # Encode the input for GP prediction
                # Create inputs that combine the numerical value with the enum value
                combined_inputs = [[x, enum_val] for x in x_grid]
                Xnum = bo.encode(combined_inputs)
                mu, sigma = bo.gp.predict(Xnum, return_std=True)

                # Plot mean and uncertainty
                ax_f.plot(x_grid, mu, color='C0', lw=2, label=r"posterior mean $\mu(x)$")
                ax_f.fill_between(x_grid, mu - sigma, mu + sigma,
                                  color='C0', alpha=0.2,
                                  label=r"uncertainty $\pm\sigma(x)$")

                # Plot acquisition function
                acq = bo.acquisition(mu, sigma)
                ax_u.plot(x_grid, acq,
                          color='limegreen', lw=2, alpha=0.6,
                          label=fr"$f({bo.acq_func})(x)$")

                # Plot selected points if available
                if x_sel is not None:
                    # Filter selected points for this enum value
                    x_sel_filtered = [x for x in x_sel if x[1] == enum_val]
                    if x_sel_filtered:
                        xs_sel = [x[0] for x in x_sel_filtered]
                        Xnum_sel = bo.encode(x_sel_filtered)
                        mu_s, sigma_s = bo.gp.predict(Xnum_sel, return_std=True)
                        acq_s = bo.acquisition(mu_s, sigma_s)
                        ax_u.scatter(xs_sel, acq_s,
                                     marker='v', color='C3', s=100,
                                     label="Selected")

            # Plot concrete samples if available
            if concrete_samples:
                # Filter samples for this enum value
                filtered_samples = [s for s in concrete_samples
                                    if s.get(f"{enum_param}_0") == enum_val]
                if filtered_samples:
                    xs = [s[f"{num_param}_0"] for s in filtered_samples]
                    ys = [s['criticality'] for s in filtered_samples]
                    ax_f.scatter(xs, ys, color='k', s=30, label='Concrete Samples')

            # Add grid and labels
            ax_f.grid(True)
            ax_u.grid(True)

            # Only add y-labels to leftmost column
            if i == 0:
                ax_f.set_ylabel("criticality")
                ax_u.set_ylabel("acquisition")

            # Add x-label to bottom row
            ax_u.set_xlabel(num_param)

        # Create a single legend for the entire figure
        handles = []
        labels = []
        for ax in axes.flatten():
            h, l = ax.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in labels:
                    handles.append(hi)
                    labels.append(li)

        fig.legend(handles, labels,
                   loc='lower center', bbox_to_anchor=(0.5, 0),
                   ncol=min(4, len(handles)), frameon=False)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for the legend
        plt.show()
        return

    # Handle case with exactly one parameter (either numerical or enum)
    if len(keys) != 1:
        print("Only 1-D plotting is supported; received", keys)
        return
    p = keys[0]

    # --- ENUM case ---
    if p in enum_parameters and p not in numerical_parameters:
        vals = enum_parameters[p][0]['values']
        N = len(vals)
        # discrete x positions 0..N-1
        xs = np.arange(N)
        # objective f at encoded value i+1
        y_obj = np.array([f([i + 1]) for i in xs])

        fig, (ax_f, ax_u) = plt.subplots(
            2, 1, sharex=True,
            figsize=(8, 6),
            gridspec_kw={'height_ratios': [3, 1]}
        )
        # plot objective
        ax_f.plot(xs, y_obj, color='gray', lw=2, label=fr"objective $f({p})$")

        # GP posterior?
        if bo is not None:
            # build raw lists of strings for encode()
            Xraw = [[v] for v in vals]
            Xnum = bo.encode(Xraw)
            mu, sigma = bo.gp.predict(Xnum, return_std=True)
            ax_f.plot(xs, mu, color='C0', lw=2, label=r"posterior mean $\mu(x)$")
            ax_f.fill_between(xs, mu - sigma, mu + sigma,
                              color='C0', alpha=0.2,
                              label=r"uncertainty $\pm\sigma(x)$")

        # concrete samples
        if concrete_samples:
            # count how many times each category was sampled
            cnt = Counter(s[p + "_0"] for s in concrete_samples)
            for cat, c in cnt.items():
                i = vals.index(cat)
                lbl = cat if c == 1 else f"{cat}({c})"
                ax_f.scatter(i, f([i + 1]), color='k', s=60, zorder=3)
                ax_f.text(i, f([i + 1]) + 0.02, lbl,
                          ha='center', va='bottom', rotation=45, fontsize=8)

        ax_f.set_ylabel("criticality")
        ax_f.set_xticks(xs)
        ax_f.set_xticklabels([f"{v}={i}" for i, v in enumerate(vals)])
        ax_f.grid(True)

        # acquisition plane
        if bo is not None:
            acq = bo.acquisition(mu, sigma)
            ax_u.plot(xs, acq, color='limegreen', lw=2, alpha=0.6,
                      label=fr"$f({bo.acq_func})(x)$")
            # selected points
            if x_sel is not None:
                # x_sel are raw lists like ['foggy']
                sel_idxs = [vals.index(x[0]) for x in x_sel]
                # recompute ucb at those
                Xnum_sel = bo.encode([[vals[i]] for i in sel_idxs])
                mu_s, sigma_s = bo.gp.predict(Xnum_sel, return_std=True)
                acq_s = bo.acquisition(mu_s, sigma_s)
                ax_u.scatter(sel_idxs, acq_s,
                             marker='v', color='C3', s=100,
                             label="Selected")

            ax_u.set_ylabel("acquisition")
            ax_u.grid(True)

        ax_u.set_xlabel(p)
        # legends
        h1, l1 = ax_f.get_legend_handles_labels()
        h2, l2 = ax_u.get_legend_handles_labels()
        ax_u.legend(h1 + h2, l1 + l2,
                    loc='upper left', bbox_to_anchor=(0, -0.3),
                    ncol=3, frameon=False)

        plt.tight_layout()
        plt.show()
        return

    # exactly one numeric (cont) parameter
    lo = numerical_parameters[p][0]['min']
    hi = numerical_parameters[p][0]['max']
    x_grid = np.linspace(lo, hi, resolution)
    y_obj = np.array([f([x]) for x in x_grid])

    fig, (ax_f, ax_u) = plt.subplots(
        2, 1, sharex=True,
        figsize=(8, 6),
        gridspec_kw={'height_ratios': [3, 1]}
    )
    ax_f.plot(x_grid, y_obj,
              color='gray', lw=2,
              label=fr"objective $f({p})$")

    if bo is not None:
        Xnum = bo.encode([[x] for x in x_grid])
        mu, sigma = bo.gp.predict(Xnum, return_std=True)
        ax_f.plot(x_grid, mu, color='C0', lw=2,
                  label=r"posterior mean $\mu(x)$")
        ax_f.fill_between(x_grid,
                          mu - sigma, mu + sigma,
                          color='C0', alpha=0.2,
                          label=r"uncertainty $\pm\sigma(x)$")

    if concrete_samples:
        xs = [s[f"{p}_0"] for s in concrete_samples]
        ys = [s['criticality'] for s in concrete_samples]
        ax_f.scatter(xs, ys, color='k', s=30, label='Concrete Samples')

    ax_f.set_ylabel("criticality")
    ax_f.grid(True)

    if bo is not None:
        acq = bo.acquisition(mu, sigma)
        ax_u.plot(x_grid, acq,
                  color='limegreen', lw=2, alpha=0.6,
                  label=fr"$f({bo.acq_func})(x)$")
        if x_sel is not None:
            xs_sel = [x[0] for x in x_sel]
            Xnum_sel = bo.encode([[x] for x in xs_sel])
            mu_s, sigma_s = bo.gp.predict(Xnum_sel, return_std=True)
            acq_s = bo.acquisition(mu_s, sigma_s)
            ax_u.scatter(xs_sel, acq_s,
                         marker='v', color='C3', s=100,
                         label="Selected")
        ax_u.set_ylabel("acquisition")
        ax_u.grid(True)

    ax_u.set_xlabel(p)
    h1, l1 = ax_f.get_legend_handles_labels()
    h2, l2 = ax_u.get_legend_handles_labels()
    ax_u.legend(h1 + h2, l1 + l2,
                loc='upper left', bbox_to_anchor=(0, -0.3),
                ncol=3, frameon=False)

    plt.tight_layout()
    plt.show()