import numpy as np
from scipy.stats import norm



class BayesianOptimizer:
    def __init__(self, param_bounds, acq_func="UCB", kappa =2.0, random_state=42):
        from skopt.learning import GaussianProcessRegressor as GP
        from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

        self.param_bounds = param_bounds
        self.rng = np.random.RandomState(random_state)
        self.acq_func = acq_func
        self.kappa = kappa

        self.dims = []
        for name, bounds in param_bounds.items():
            if isinstance(bounds, list):
                # continious
                kind, meta = 'categorical', bounds

            else:
                # assume a tuple (low, high)
                low, high = bounds
                # both endpoints are ints → integer
                if isinstance(low, int) and isinstance(high, int):
                    kind, meta = 'int', bounds
                else:
                    kind, meta = 'real', bounds

            self.dims.append((name, kind, meta))

        amplitude = ConstantKernel(1.0, (1e-2, 1e2))
        matern = Matern(length_scale=10.0,
                        length_scale_bounds=(1.0, 100.0),
                        nu=2.5)
        noise = WhiteKernel(noise_level=1e-3,
                            noise_level_bounds=(1e-5, 1e-1))
        kernel = amplitude * matern + noise

        # Fit GP with more restarts for kernel optimization
        self.gp = GP(kernel=kernel,
                     normalize_y=True,
                     n_restarts_optimizer=10,
                     random_state=random_state)

        self.X_train = []
        self.y_train = []

    def fit(self, X_raw, y_raw):  # store raw training data
        X_num = self.encode(X_raw)
        self.gp.fit(X_num, np.asarray(y_raw))
        self.X_train = list(X_raw)  # list of lists of parameter values
        self.y_train = list(y_raw)  # list of criticality floats

    def encode(self, X_raw): # list of lists to numeric array
        rows = []
        for point in X_raw:
            r = []
            for (name, kind, meta), raw in zip(self.dims, point):

                if kind in ('real', 'int'):
                    r.append(raw)
                else: # only for enums
                    r.append(meta.index(raw))
            rows.append(r)
        return np.array(rows)

    def sample_candidates(self, n_cand): #umifotmly sample n points in original parameter interval
        cand = []
        for _ in range(n_cand):
            pt = []
            for name, kind, meta in self.dims:
                if kind == 'real':
                    low, high = meta
                    pt.append(self.rng.uniform(low, high))
                elif kind == 'int':
                    low, high = meta
                    # draw integer in [lo..hi]
                    pt.append(int(self.rng.randint(low, high + 1)))
                else:
                    pt.append(self.rng.choice(meta))
            cand.append(pt)
        return cand

    def acquisition(self, mu, sigma):
        if self.acq_func == "UCB":
            return mu + self.kappa * sigma

        # need at least one observed y to compute improvement over
        best_y = np.max(self.y_train)

        if self.acq_func == "EI":
            # Expected Improvement
            with np.errstate(divide="ignore", invalid="ignore"):
                z = (mu - best_y) / sigma
                ei = (mu - best_y) * norm.cdf(z) + sigma * norm.pdf(z)
                ei[sigma == 0.0] = 0.0
            return ei

        if self.acq_func == "PI":
            # Probability of Improvement
            with np.errstate(divide="ignore", invalid="ignore"):
                z = (mu - best_y) / sigma
                pi = norm.cdf(z)
            return pi

        raise NotImplementedError(f"Acquisition function '{self.acq_func}' not implemented.")

    # compute 200 candidates and select top-K scenarios
    def propose(self, K, n_candidates=200, penalise=True, penalty_scale=1.0):

        Xcand = self.sample_candidates(n_candidates) # sample candidate pool in raw space
        Xnum = self.encode(Xcand) #encode candidates and predict

        # GP predict
        mu, sigma = self.gp.predict(Xnum, return_std=True)

        # computing acquisition
        acq = self.acquisition(mu, sigma)

        """ IF YOU WANT TO LOG ALL OF THE CANDIDATES
        # logging every candidate
        for x_raw, y_pred in zip(Xcand, mu):
            # turn the raw list into a Python tuple for nicer output:
            tup = tuple(x_raw)
            print(f"Generated point {tup} → predicted criticality {y_pred:.3f}")
        """
        # select K with optional penalisation
        acq_work = acq.copy()
        selected_idxs = []

        for _ in range(K):
            idx = int(np.argmax(acq_work))
            selected_idxs.append(idx)

            if penalise:
                dists = np.linalg.norm(Xnum - Xnum[idx], axis=1)
                penalty = np.exp(- (dists ** 2) / (2 * penalty_scale ** 2))
                acq_work *= (1.0 - penalty)

        # prepare final outputs
        X_sel = [Xcand[i] for i in selected_idxs]
        y_sel = mu[selected_idxs]

        # log selected
        for x_raw, y_pred in zip(X_sel, y_sel):
            params_str = ", ".join(
                f"{name}={val!r}" for (name, _, _), val in zip(self.dims, x_raw)
            )
            print(f" -> Selected {params_str} with pred. criticality {y_pred:.3f}")

        return X_sel, y_sel # list of K raw vectors and predicted items
