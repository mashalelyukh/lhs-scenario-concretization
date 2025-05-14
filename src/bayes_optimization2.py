import numpy as np
from utils import ask_yes_no


class BayesianOptimizer:
    def __init__(self, param_bounds, acq_func="EI", random_state=42):
        from skopt.space import Real, Categorical
        from skopt.learning import GaussianProcessRegressor as GP
        from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

        # Save metadata
        self.param_bounds = param_bounds
        self.random_state = random_state
        # Dimensions list for sampling
        self.dims = []
        for name, bounds in param_bounds.items():
            if isinstance(bounds, list):
                # Categorical: sample via index later
                self.dims.append((name, 'categorical', bounds))
            else:
                # Continuous: (name, 'real', (low, high))
                self.dims.append((name, 'real', bounds))

        # Initialize a GP surrogate with limited hyperparam tuning for speed
        kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(1e-6)
        self.gp = GP(kernel=kernel, normalize_y=True,
                     n_restarts_optimizer=2, random_state=random_state)

    def fit(self, X, y):
        """
        X: list of lists of raw param values (floats or strings)
        y: list of criticality floats
        """
        # Store raw training data
        self.X_train = X
        self.y_train = np.asarray(y)

    def _encode(self, X_raw):
        """
        Convert raw sample list-of-lists into numeric array for GP.
        """
        X_num = []
        for point in X_raw:
            row = []
            for (name, kind, meta), raw in zip(self.dims, point):
                if kind == 'real':
                    row.append(raw)
                else:  # 'categorical'
                    # map category to integer index
                    cats = meta
                    idx = cats.index(raw)
                    row.append(idx)
            X_num.append(row)
        return np.array(X_num)

    def _sample_candidates(self, n_samples):
        """
        Uniformly sample n_samples points in original space.
        """
        cand = []
        rng = np.random.RandomState(self.random_state)
        for _ in range(n_samples):
            pt = []
            for name, kind, meta in self.dims:
                if kind == 'real':
                    low, high = meta
                    pt.append(rng.uniform(low, high))
                else:
                    pt.append(rng.choice(meta))
            cand.append(pt)
        return cand

    def propose(self, K, C, n_candidates=200):
        """
        Fit GP on training data, sample candidates, predict, and select top-K.
        Graceful fallback if C out of reach.
        Returns list of K raw vectors and their predicted means.
        """
        # 1) Fit GP on stored training data
        Xnum_train = self._encode(self.X_train)
        self.gp.fit(Xnum_train, self.y_train)

        # 2) Sample candidate pool in raw space
        Xcand = self._sample_candidates(n_candidates)
        # 3) Encode candidates and predict
        Xnum = self._encode(Xcand)
        mu, sigma = self.gp.predict(Xnum, return_std=True)
        ucb = mu + 2 * sigma

        # 4) Check feasibility
        max_ucb = ucb.max()
        if C > max_ucb:
            print(f"Requested C={C:.2f} exceeds max achievable μ+2σ={max_ucb:.2f}.")
            if ask_yes_no(f"Lower C to {max_ucb:.2f}? (yes/no)"):
                C = max_ucb
            else:
                print("Proceeding with best-effort: top-K by mean μ.")
                idx = np.argsort(-mu)[:K]
                return [Xcand[i] for i in idx], mu[idx]

        # 5) Select those above threshold
        sel = [i for i, m in enumerate(mu) if m >= C]
        if len(sel) < K:
            print(f"Only {len(sel)} ≥ {C:.2f}, filling to {K} with top μ.")
            ranked = list(np.argsort(-mu))
            for i in ranked:
                if i not in sel and len(sel) < K:
                    sel.append(i)
        else:
            sel = sel[:K]

        X_sel = [Xcand[i] for i in sel]
        y_sel = mu[sel]
        print(f"Proposed {len(X_sel)} scenarios (μ): {np.round(y_sel, 3)}")
        return X_sel, y_sel
