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
    mapped_values = {}
    for i, key in enumerate(parameters.keys()):
        spec = parameters[key]

        if 'values' in spec:  # enum/list case
            num_options = len(spec['values'])
            index = int(np.floor(sample[i] * num_options))  # scales normalized sample to index space
            index = min(index, num_options - 1)  # upper bound safety
            concrete_value = spec['values'][index]
        else:  # Numerical case
            min_val = spec['min']
            max_val = spec['max']
            concrete_value = min_val + sample[i] * (
                    max_val - min_val)  # scales the normalized value to the concrete range
            if spec['type_annotation'] in ['int', 'uint']:
                concrete_value = int(round(concrete_value))
            else:
                concrete_value = float(concrete_value)

        mapped_values[key] = concrete_value

    return mapped_values

#FOR FUTURE GEAR CHECKERR
def constraint_checker(mapped_values):
    """
    Check if the concrete parameter values satisfy all the constraints.

    This function can be extended to include domain-specific rules or inter-parameter checks.
    Currently, it simply returns True.
    """
    # For example, one might enforce additional constraints like:
    # if mapped_values['gear'] < 0 and mapped_values['speed'] > 100:
    #     return False
    return True


def adjust_sampling_strategy(mapped_values):
    # Placeholder for adjusting the adaptive sampling strategy.

    # might log the rejected sample, adjust the random seed, or modify the sampling
    # grid based on which regions are rejected. For now, it simply prints the sample.
    print("Rejected sample:", mapped_values)


def generate_concrete_parameter_samples(num_samples, numerical_parameters, enum_parameters):
    flat_parameters = flatten_parameters(numerical_parameters, enum_parameters)

    # generate initial candidate points using adaptive LHS
    normalized_samples = adaptive_lhs_sampler(num_samples, flat_parameters)

    concrete_samples = []
    for sample in normalized_samples:
        # map the normalized sample to concrete parameter values
        mapped_values = parameter_mapper(sample, flat_parameters)

        # check if the mapped values satisfy all constraints
        if constraint_checker(mapped_values):
            concrete_samples.append(mapped_values)
        else:
            # Optionally log the rejected sample and provide feedback for adapting the LHS
            adjust_sampling_strategy(mapped_values)

    # Optionally run additional iterations to improve the sample set
    return concrete_samples
