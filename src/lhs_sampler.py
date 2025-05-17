import numpy as np


def normalize_ranges(numerical_parameters):
    for specs in numerical_parameters.values():
        for spec in specs:
            if 'min' in spec and 'max' in spec and spec['min'] > spec['max']:
                spec['min'], spec['max'] = spec['max'], spec['min']


def flatten_parameters(numerical_parameters, enum_parameters):  # flattens parameters so each occurrence gets its key.
    flat = {}
    for key, specs in numerical_parameters.items():
        for idx, spec in enumerate(specs):
            composite_key = f"{key}_{idx}"  # creates keys like "speed_0", "speed_1", etc.
            flat[composite_key] = spec
    for key, specs in enum_parameters.items():
        for idx, spec in enumerate(specs):
            composite_key = f"{key}_{idx}"
            flat[composite_key] = spec
    return flat


# performs lhs on continuous parameters (floats) and
# controlled random sampling on discrete ones.
# for M discrete possible values and N scenarios:
# if M >= N, sample N unique values without replacement
# if M < N, sample each value N/M times, random remainder
def adaptive_lhs_sampler(num_samples, parameter_specs):
    num_parameters = len(parameter_specs)
    samples = np.empty((num_samples, num_parameters))  # empty array to hold the samples

    discrete_axes = []
    continuous_axes = []
    discrete_domains = {}

    for j, spec in enumerate(parameter_specs.values()):
        if 'values' in spec:
            # enum
            discrete_axes.append(j)
            discrete_domains[j] = spec['values']
        elif spec.get('type_annotation') in ['int', 'uint']:
            # integer range
            discrete_axes.append(j)
            min_val, max_val = int(spec['min']), int(spec['max'])
            discrete_domains[j] = list(range(min_val, max_val + 1))
        else:
            continuous_axes.append(j)
    # Latin Hypercube for continuous dimensions
    for j in continuous_axes:
        cuts = np.linspace(0, 1, num_samples + 1)
        lower, upper = cuts[:-1], cuts[1:]
        col = np.random.uniform(lower, upper)
        np.random.shuffle(col)
        samples[:, j] = col

        # Controlled sampling for discrete dimensions
    discrete_choices = {}
    for j in discrete_axes:
        domain = discrete_domains[j]
        M, N = len(domain), num_samples
        if M >= N:
            chosen = list(np.random.choice(domain, N, False))
        else:
            base = N // M
            rem = N % M
            pool = domain * base
            pool += list(np.random.choice(domain, rem, False))
            np.random.shuffle(pool)
            chosen = pool
        discrete_choices[j] = chosen
        # fill samples with dummy uniform within bin centers (optional)
        u_vals = []
        for idx_bin in range(len(chosen)):
            bin_idx = domain.index(chosen[idx_bin])
            u_vals.append((bin_idx + .5) / M)
        samples[:, j] = u_vals

    return samples, discrete_choices


def parameter_mapper(sample, parameters):
    # foats -> linear scaling
    # ints/uints -> equal-sized bins through floor mapping

    mapped_values = {}
    keys = list(parameters.keys())

    for i, key in enumerate(keys):
        spec = parameters[key]
        u = sample[i]
        u = float(u)

        # enums
        if 'values' in spec:
            domain = spec['values']
            n_opts = len(domain)
            idx = min(int(np.floor(u * n_opts)), n_opts - 1)
            concrete = domain[idx]

        # int/uint
        elif spec['type_annotation'] in ['int', 'uint']:
            min_val = int(spec['min'])
            max_val = int(spec['max'])
            domain = list(range(min_val, max_val + 1))
            n_opts = len(domain)
            idx = min(int(np.floor(u * n_opts)), n_opts - 1)
            concrete = domain[idx]

        # float
        else:
            min_val = float(spec['min'])
            max_val = float(spec['max'])
            concrete = min_val + u * (max_val - min_val)

        mapped_values[key] = concrete

    return mapped_values


def adjust_sampling_strategy(mapped_values):
    print("Rejected sample:", mapped_values)


def generate_concrete_parameter_samples(num_samples, numerical_parameters, enum_parameters):
    normalize_ranges(numerical_parameters)
    flat_parameters = flatten_parameters(numerical_parameters, enum_parameters)
    normalized_samples, discrete_map = adaptive_lhs_sampler(num_samples, flat_parameters)
    concrete_samples = []

    for i in range(num_samples):
        sample = normalized_samples[i]
        mapped = parameter_mapper(sample, flat_parameters)
        # override discrete dims with precalculated
        for j, choices in discrete_map.items():
            key = list(flat_parameters.keys())[j]
            mapped[key] = choices[i]
        concrete_samples.append(mapped)
    return concrete_samples

    # for sample in normalized_samples:
    # map the normalized sample to concrete parameter values
    #   mapped_values = parameter_mapper(sample, flat_parameters)
    # concrete_samples.append(mapped_values)
    # #optional(future): run additional iterations to improve the sample set
    # return concrete_samples
