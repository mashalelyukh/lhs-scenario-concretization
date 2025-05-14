#unnecessary file
def replace_range_with_value(line, span, concrete_value, param_spec):
    """
    Replaces the range expression at the given span in the line with the concrete value.
    Handles units and string quoting for enums.
    """
    # Handle formatting depending on parameter type
    if param_spec.get('enum'):
        replacement = f'"{concrete_value}"'
    else:
        if param_spec.get('unit'):
            replacement = f"{concrete_value}{param_spec['unit']}"
        else:
            replacement = str(concrete_value)

    # Determine replacement pattern based on type
    if param_spec['type'].startswith('enum'):
        # Example: weatherCondition : ["sunny", "rainy", "foggy"] → weatherCondition : "sunny"
        #pass  # no special symbols here, direct replacement
        replacement = f"{param_spec['param']} == {replacement}"
    elif param_spec['type'] in ['in', 'colon', 'call']:
        replacement = f"{param_spec['param']} == {replacement}" if param_spec['type'] == 'in' else f"{param_spec['param']}: {replacement}"
    #elif param_spec['type'] == 'call':
      #  replacement = f"{param_spec['param']}({replacement}"
       # if param_spec.get('extras'):
        #    replacement += f", {param_spec['extras']})"
        #else:
         #   replacement += ")"



    if param_spec['type'].startswith('enum'):
        replacement = f"{param_spec['param']} == {replacement}"
    elif param_spec['type'] == 'in':
        replacement = f"{param_spec['param']} == {replacement}"
    elif param_spec['type'] == 'colon':
        replacement = f"{param_spec['param']}:{replacement}"
    elif param_spec['type'] == 'call':
        if param_spec.get('extras'):  # If there are extras like ", at: end"
            replacement = f"{param_spec['param']}({replacement}, {param_spec['extras']})"
        else:
            replacement = f"{param_spec['param']}({replacement})"



    else:
        raise ValueError(f"Unknown parameter type: {param_spec['type']}")

    # Replace the text at the given span:
    start, end = span
    new_line = line[:start] + replacement + line[end:]
    return new_line



""" from lrk_parser rangeTransformer:
    def extract_extras(self, node):
        if hasattr(node, 'children'):
            #return ''.join(str(child) for child in node.children).strip()
        #return str(node).strip()
            return ' '.join(str(child) for child in node.children)
        return str(node)
"""



#old bayesian_opt:


import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm

class BayesianOptimizer:
    def init(self, param_bounds):
        """
        param_bounds: dict of {param_name: (min, max)}, all numerical/encoded.
        """
        self.param_bounds = param_bounds
        # Matern kernel + white noise, normalized y
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(1e-6)
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

    def fit(self, X, y):
        """X: array-like [n_samples, n_dims], y: [n_samples]"""
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.gp.fit(self.X, self.y)

    def scale_to_bounds(self, U):
        """Scale unit-cube samples U to your param_bounds."""
        mins = np.array([b[0] for b in self.param_bounds.values()])
        maxs = np.array([b[1] for b in self.param_bounds.values()])
        return mins + (maxs - mins) * U

    def expected_improvement(self, Xcand, C, xi=0.01):
        """Compute EI(Xcand) = E[max(0, f(x)-C)], for candidate points."""
        mu, sigma = self.gp.predict(Xcand, return_std=True)
        sigma = sigma.reshape(-1, 1)
        with np.errstate(divide='ignore'):
            imp = mu - C - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma.flatten() == 0.0] = 0.0
        return ei.flatten()

    def propose(self, K, C, n_candidates=5000):
        """
        Main entry:
          - Check if C is reachable;
          - If not, offer fallback;
          - Otherwise return (X_new, y_pred) of size K.
        """
        # 1) Quick UCB scan to find max possible
        max_ucb, max_mu = self.estimate_max_ucb(n_candidates)
        if C > max_ucb:
            print(f"\nThreshold C={C:.2f} is above any μ+2σ (max≈{max_ucb:.2f}).")
            from main import ask_yes_no
            if ask_yes_no(f"Lower target to {max_ucb:.2f}?"):
                C = max_ucb
            else:
                print("Proceeding with best-effort (top-K by μ).")
                X_down, y_down = self.best_effort(K, n_candidates)
                return X_down, y_down

        # 2) Otherwise run EI acquisition
        X_new, ei_vals = self.acquire_ei(K, C, n_candidates)
        y_new = self.gp.predict(X_new)
        print(f"\nProposed {K} new scenarios (predicted criticality ≥ {C:.2f}):")
        print(np.round(y_new, 3), "\n")
        return X_new, y_new

    def estimate_max_ucb(self, n_samples):
        """Sample n_samples random points, return max(μ+2σ) and max(μ)."""
        U = np.random.rand(n_samples, len(self.param_bounds))
        Xc = self.scale_to_bounds(U)
        mu, sigma = self.gp.predict(Xc, return_std=True)
        ucb = mu + 2 * sigma
        return ucb.max(), mu.max()

    def acquire_ei(self, K, C, n_samples):
        """Sample n_samples, compute EI, return top-K points."""
        U = np.random.rand(n_samples, len(self.param_bounds))
        Xc = self.scale_to_bounds(U)
        ei = self.expected_improvement(Xc, C)
        idx = np.argsort(-ei)[:K]
        return Xc[idx], ei[idx]

    def best_effort(self, K, n_samples):
        U = np.random.rand(n_samples, len(self.param_bounds))
        Xc = self.scale_to_bounds(U)
        mu = self.gp.predict(Xc)
        idx = np.argsort(-mu)[:K]
        return Xc[idx], mu[idx]


#main after 87:

    # After you have:
    #   concrete_samples: list of dicts with scenario["params"] = dict of values
    #   and scenario["criticality"] labels

    # 1) Build X (list of parameter-vectors) and y
    param_names = list(concrete_samples[0]["params"].keys())
    param_bounds = {
        name: (min_val, max_val)
        for name, (min_val, max_val) in your_extracted_bounds.items()
    }
    X = [
        [scenario["params"][n] for n in param_names]
        for scenario in concrete_samples
    ]
    y = [scenario["criticality"] for scenario in concrete_samples]

    # 2) Ask user for K and C
    K = get_int("How many new scenarios to generate? (default is 2)", 2)
    C = get_float("Desired minimum criticality (0–1)? ", 0.0, 1.0)

    # 3) Run BO
    bo = BayesianOptimizer(param_bounds)
    bo.fit(X, y)
    new_X, new_y = bo.propose(K, C)

    # 4) Inject new_X/new_y into your concretization pipeline




#previous main:
#base_dir = os.path.dirname(os.path.dirname(__file__))
    #file_path = os.path.join(base_dir, "scenarios_input", "ex2.osc")
    #numerical_parameters, enum_parameters = extract_parameters(file_path)
    """
    output_dir = os.path.join(base_dir, "scenarios_output")
    clear_output_folder(output_dir)
    numerical_parameters, enum_parameters = extract_parameters(file_path)
    print("Numerical Parameters:")
    for name, info in numerical_parameters.items():
        print(f"  {name}: {info}")
    print("\n Enum Parameters:")
    for enum_name, values in enum_parameters.items():
        print(f"  {enum_name}: {values}")
    num_samples = 10
    concrete_samples = generate_concrete_parameter_samples(num_samples, numerical_parameters, enum_parameters)
    print("\n Concrete Parameter Samples using Adaptive LHS:")
    for i, sample in enumerate(concrete_samples, start=1):
        print(f"  Sample {i}: {sample}")
    flat_parameters = flatten_parameters(numerical_parameters, enum_parameters)
    # Concretize scenarios and write output files
    concretize_scenario(
        file_path,
        output_dir,
        concrete_samples,
        flat_parameters
    )
    print(f"\n Concrete scenarios generated in '{output_dir}'")
    """

    #from main:
    # 1) Build param_bounds from what you extracted earlier
    #    (same logic you used to feed your LHS sampler)
    param_bounds = {}
    for name, info_list in numerical_parameters.items():
        info = info_list[0]
        lb, ub = info["min"], info["max"]
        param_bounds[name] = (lb, ub)

        # param_bounds[name] = (info["min"], info["max"])
    # for name, cats in enum_parameters.items():
    # param_bounds[name] = cats  # list → treated as categorical

    for name, info_list in enum_parameters.items():
        info = info_list[0]
        cats = info["values"]  # e.g. ["sunny","rainy","foggy"]
        param_bounds[name] = cats

    # 2) Prepare X, y for BO
    param_names = list(param_bounds.keys())















    #main from 103 till end:
    # 1) How many were there already?
    existing_files = sorted(os.listdir(output_dir))
    N = len(existing_files)

    # 2) Prepare for single‐sample concretization
    base_name, ext = os.path.splitext(os.path.basename(file_path))
    # flat_parameters was computed earlier via flatten_parameters(...)
    # param_names is list(param_bounds.keys())

    for offset, (x_vec, y_pred) in enumerate(zip(new_X, new_y), start=1):
        # build sample dict:
        sample = {name: x_vec[j] for j, name in enumerate(param_names)}

        # snapshot before generation
        before = set(os.listdir(output_dir))

        # 3) Generate exactly one new .osc file
        concretize_scenario(
            file_path,
            output_dir,
            [sample],  # one‐element list
            flat_parameters
        )

        # 4) Find which file appeared
        after = set(os.listdir(output_dir))
        new_files = after - before
        if len(new_files) != 1:
            raise RuntimeError("Expected exactly one new file, found: " + str(new_files))
        old_name = new_files.pop()

        # 5) Rename so indexing continues
        new_index = N + offset
        new_name = f"{base_name}_{new_index}{ext}"
        os.rename(
            os.path.join(output_dir, old_name),
            os.path.join(output_dir, new_name)
        )

        # 6) Console feedback
        print(f"for {new_name} the expected criticality value is {y_pred:.3f}")
