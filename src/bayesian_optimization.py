import numpy as np
from utils import ask_yes_no

class BayesianOptimizer:
    def __init__(self, param_bounds, acq_func="EI", random_state=42):
        from skopt import Optimizer
        from skopt.space import Real, Categorical
        """
        param_bounds: dict mapping parameter names to either:
          - tuple (min, max) for continuous parameters
          - list of categories for enum parameters
        """
        self.param_names = list(param_bounds.keys())
        dims = []
        for name, bounds in param_bounds.items():
            if isinstance(bounds, list):
                # Categorical parameter
                dims.append(Categorical(bounds, name=name))
            elif isinstance(bounds, tuple) and len(bounds) == 2:
                # Continuous parameter
                dims.append(Real(bounds[0], bounds[1], name=name))
            else:
                raise ValueError(f"Unsupported bounds for '{name}': {bounds}")

        # Initialize the skopt Optimizer with a Gaussian Process surrogate
        self.opt = Optimizer(
            dimensions=dims,
            base_estimator="GP",
            acq_func=acq_func,
            random_state=random_state
        )

    def fit(self, X, y):
        """
        Ingest existing data for the surrogate model.
        X: list of lists, each inner list is parameter values in the order of self.param_names
        y: list of criticality values
        """
        self.opt.tell(X, y)

    def propose(self, K, C, n_candidates=1000):
        """
        Propose K new scenarios with predicted criticality ≥ C.
        If C is out of reach (above max μ+2σ), offer graceful fallback.
        Returns:
          - X_new: list of new parameter vectors
          - y_pred: array of predicted criticalities for X_new
        """
        # 1) Generate a pool of candidate points
        Xcand = self.opt.ask(n_points=n_candidates)

        # 1b) Transform to numeric array for the GP surrogate
        Xnum = np.array(self.opt.space.transform(Xcand))

        # 2) Query the internal GP surrogate for mean and std
        gp = self.opt.base_estimator_
        #mu, sigma = gp.predict(Xcand, return_std=True)
        mu, sigma = gp.predict(Xnum, return_std=True)
        ucb = mu + 2 * sigma
        max_ucb = ucb.max()

        # 3) Check threshold feasibility
        if C > max_ucb:
            print(f"Requested C={C:.2f} exceeds max achievable μ+2σ={max_ucb:.2f}.")
            if ask_yes_no(f"Lower C to {max_ucb:.2f}? (yes/no) "):
                C = max_ucb
            else:
                print("Proceeding with best-effort: selecting top-K by mean μ.")
                idx = np.argsort(-mu)[:K]
                X_sel = [Xcand[i] for i in idx]
                y_sel = mu[idx]
                return X_sel, y_sel

        # 4) Filter candidates meeting the threshold
        valid_idx = [i for i, m in enumerate(mu) if m >= C]
        if len(valid_idx) >= K:
            selected_idx = valid_idx[:K]
        else:
            print(f"Only {len(valid_idx)} candidates ≥ {C:.2f}. Filling up to {K} with top means.")
            # fill remaining slots with highest-μ candidates
            ranked = list(np.argsort(-mu))
            for idx in ranked:
                if idx not in valid_idx and len(valid_idx) < K:
                    valid_idx.append(idx)
            selected_idx = valid_idx

        # 5) Assemble final proposals
        X_sel = [Xcand[i] for i in selected_idx]
        y_sel = mu[selected_idx]
        print(f"Proposed {len(X_sel)} scenarios with predicted criticalities: {np.round(y_sel,3)}")
        return X_sel, y_sel
