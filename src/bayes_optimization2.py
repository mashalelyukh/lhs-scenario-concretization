import numpy as np
from utils import ask_yes_no


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

        #kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(1e-6)
        #n_restarts_optimzer - to be found
        #self.gp = GP(kernel, True, n_restarts_optimizer=2, random_state=random_state)

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

#???????????????????????????????????????
    def acquisition(self, mu, sigma):
        #if self.acq_func == "TS":
         #   return None
        #if
        if self.acq_func == "UCB":
            return mu + self.kappa * sigma
        else:
            raise NotImplementedError("Only UCB implemented.")
#???????????????????????????????????????

    def propose(self, K, n_candidates=200):  #select top-K scenarios
        Xcand = self.sample_candidates(n_candidates) # sample candidate pool in raw space
        Xnum = self.encode(Xcand) #encode candidates and predict

        # the UCB route:
        mu, sigma = self.gp.predict(Xnum, return_std=True)
        acq = self.acquisition(mu, sigma)

        """ IF YOU WANT TO LOG ALL OF THE CANDIDATES
        # logging every candidate
        for x_raw, y_pred in zip(Xcand, mu):
            # turn the raw list into a Python tuple for nicer output:
            tup = tuple(x_raw)
            print(f"Generated point {tup} → predicted criticality {y_pred:.3f}")
        """
        # working copy we can zero‐out
        acq_work = acq.copy()

        selected_idxs = []
        # define your “no-go” radius in the encoded space:
        # you’ll need to pick a number that makes sense for your parameters
        radius = 0.05  # ← tune this!

        for _ in range(K):
            # pick the next best point
            idx = int(np.argmax(acq_work))
            selected_idxs.append(idx)

            # compute distances from this pick to all candidates
            dists = np.linalg.norm(Xnum - Xnum[idx], axis=1)

            # penalize all points within `radius`, including itself
            acq_work[dists <= radius] = -np.inf

            # if we’ve excluded everyone, “reset” so we can refill
            if np.all(acq_work == -np.inf):
                acq_work = acq.copy()

                # build your batch
        X_sel = [Xcand[i] for i in selected_idxs]
        y_sel = mu[selected_idxs]

        for x_raw, y_pred in zip(X_sel, y_sel):
            params_str = ", ".join(f"{name}={val!r}"
                                   for (name, _, _), val in zip(self.dims, x_raw))
            print(f"Generated point with {params_str} → predicted criticality {y_pred:.3f}")

        """
        # logging every candidate
        for x_raw, y_pred in zip(Xcand, mu):
            # turn the raw list into a Python tuple for nicer output:
            tup = tuple(x_raw)
            print(f"Generated point {tup} → predicted criticality {y_pred:.3f}")
        
        
        # 4) Batch‐selection for diversity: random‐subset from the top-M
        M = min(5 * K, len(acq))  # shortlist size
        ranked = np.argsort(-acq)[:M]  # indices of top-M by acq
        chosen = self.rng.choice(ranked, size=K, replace=False)
        X_sel = [Xcand[i] for i in chosen]
        y_sel = mu[chosen]
        
        # for controlling the points
        for x_raw, y_pred in zip(X_sel, y_sel):
            params_str = ", ".join(f"{name}={val!r}"
                                   for (name, _, _), val in zip(self.dims, x_raw))
            print(f"Generated point with {params_str} → predicted criticality {y_pred:.3f}")
            """

        return X_sel, y_sel  # list of K raw vectors and predicted items
