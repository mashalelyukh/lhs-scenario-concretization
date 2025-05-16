import numpy as np


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


def adaptive_lhs_sampler(num_samples, parameter_specs):
    num_parameters = len(parameter_specs)
    samples = np.empty((num_samples, num_parameters))  # empty array to hold the samples

    for j in range(num_parameters):  # for each parameter/dimension, generate LHS samples
        cuts = np.linspace(0, 1, num_samples + 1)  # creates equally spaced intervals
        lower_bounds = cuts[:num_samples]
        upper_bounds = cuts[1:num_samples + 1]

        col_samples = np.random.uniform(lower_bounds, upper_bounds,
                                        size=num_samples)  # samples uniformly within each interval

        np.random.shuffle(col_samples)  # ensures random pairing across dimensions
        samples[:, j] = col_samples

    return samples


def parameter_mapper(sample, parameters):
    #foats -> linear scaling
    #ints/uints -> equal-sized bins through floor mapping

    mapped_values = {}
    keys = list(parameters.keys())

    for i, key in enumerate(keys):
        spec = parameters[key]
        u = sample[i]

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
    flat_parameters = flatten_parameters(numerical_parameters, enum_parameters)

    # generate initial candidate points using adaptive LHS
    normalized_samples = adaptive_lhs_sampler(num_samples, flat_parameters)

    concrete_samples = []
    for sample in normalized_samples:
        # map the normalized sample to concrete parameter values
        mapped_values = parameter_mapper(sample, flat_parameters)
        concrete_samples.append(mapped_values)
    # optional(future): run additional iterations to improve the sample set
    return concrete_samples
