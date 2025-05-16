import numpy as np
from utils import ask_yes_no


class BayesianOptimizer:
    def __init__(self, param_bounds, acq_func="EI", random_state=42):
        self.X_train = None
        self.y_train = None
        from skopt.space import Real, Categorical
        from skopt.learning import GaussianProcessRegressor as GP
        from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

        self.param_bounds = param_bounds
        self.random_state = random_state

        self.dims = []
        for name, bounds in param_bounds.items():
            if isinstance(bounds, list):
                self.dims.append((name, 'categorical', bounds))  # discrete params
            else:
                self.dims.append((name, 'real', bounds))  # continious params

        kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(1e-6)
        self.gp = GP(kernel=kernel, normalize_y=True,
                     n_restarts_optimizer=2, random_state=random_state)

    def fit(self, X, y):  # store raw training data
        self.X_train = X  # list of lists of parameter values
        self.y_train = np.asarray(y)  # list of criticality floats

    def encode(self, X_raw): # list of lists to numeric array
        X_num = []
        for point in X_raw:
            row = []
            for (name, kind, meta), raw in zip(self.dims, point):
                if kind == 'real':
                    row.append(raw)
                else:  # discrete
                    # map discrete to integer index
                    cats = meta
                    idx = cats.index(raw)
                    row.append(idx)
            X_num.append(row)
        return np.array(X_num)

    def sample_candidates(self, n_samples): #umifotmly sample n points in original parameter interval
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

    def propose(self, K, C, n_candidates=200):  #fitting GP on data, select top-K
        Xnum_train = self.encode(self.X_train)
        self.gp.fit(Xnum_train, self.y_train)

        Xcand = self.sample_candidates(n_candidates) # sample candidate pool in raw space
        Xnum = self.encode(Xcand) #encode candidates and predict
        mu, sigma = self.gp.predict(Xnum, return_std=True)
        ucb = mu + 2 * sigma

        # check feasibility
        max_ucb = ucb.max()
        if C > max_ucb:
            print(f"Requested C={C:.2f} exceeds max achievable μ+2σ={max_ucb:.2f}.")
            if ask_yes_no(f"Lower C to {max_ucb:.2f}? (yes/no)"):
                C = max_ucb
            else:
                print("Proceeding with best-effort: top-K by mean μ.")
                idx = np.argsort(-mu)[:K]
                return [Xcand[i] for i in idx], mu[idx]

        # select above threshold
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
        return X_sel, y_sel #list of K raw vectors and predicted items
