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

        # initializes skopt Optimizer with a Gaussian Process surrogate
        self.opt = Optimizer(
            dimensions=dims,
            base_estimator="GP",
            acq_func=acq_func,
            random_state=random_state
        )

    def fit(self, X, y):
        self.opt.tell(X, y)

    def propose(self, K, C, n_candidates=1000):
        Xcand = self.opt.ask(n_points=n_candidates)

        Xnum = np.array(self.opt.space.transform(Xcand))
        gp = self.opt.base_estimator_
        #mu, sigma = gp.predict(Xcand, return_std=True)
        mu, sigma = gp.predict(Xnum, return_std=True)
        ucb = mu + 2 * sigma
        max_ucb = ucb.max()
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

        X_sel = [Xcand[i] for i in selected_idx]
        y_sel = mu[selected_idx]
        print(f"Proposed {len(X_sel)} scenarios with predicted criticalities: {np.round(y_sel,3)}")
        return X_sel, y_sel
