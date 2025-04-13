import numpy as np


#def adaptive_lhs_sampler(num_samples, parameter_specs):
def adaptive_lhs_sampler(num_samples, numerical_parameters):
    """
    Latin Hypercube Sampling: generate points in the [0,1] space.

    For each parameter (dimension), the interval [0,1] is divided into equal subintervals.
    A random value is drawn from each subinterval and the resulting values are shuffled.
    """
    num_parameters = len(numerical_parameters)
    # Initialize an empty array to hold the samples
    samples = np.empty((num_samples, num_parameters))

    # For each parameter/dimension, generate LHS samples
    for j in range(num_parameters):
        # Create equally spaced intervals
        cuts = np.linspace(0, 1, num_samples + 1)
        lower_bounds = cuts[:num_samples]
        upper_bounds = cuts[1:num_samples + 1]
        # Sample uniformly within each interval
        col_samples = np.random.uniform(lower_bounds, upper_bounds, size=num_samples)
        # Shuffle the order to ensure random pairing across dimensions
        np.random.shuffle(col_samples)
        samples[:, j] = col_samples

    return samples


def parameter_mapper(sample, numerical_parameters):
    """
    Map a normalized sample (list/array of values in [0,1]) to concrete parameter values.

    Each normalized value is scaled using the formula:
      concrete_value = min + normalized_value * (max - min)
    The value is then cast to the appropriate type based on type_annotation.
    """
    mapped_values = {}
    # Ensure a consistent parameter order by iterating over the keys sorted (or in insertion order)
    for i, key in enumerate(numerical_parameters.keys()):
        spec = numerical_parameters[key]
        min_val = spec['min']
        max_val = spec['max']

        # Scale the normalized value to the concrete range
        concrete_value = min_val + sample[i] * (max_val - min_val)

        # Apply type conversion based on type_annotation
        if spec['type_annotation'] in ['int', 'uint']:
            concrete_value = int(round(concrete_value))
        else:
            concrete_value = float(concrete_value)

        mapped_values[key] = concrete_value
    return mapped_values


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
    """
    Placeholder for adjusting the adaptive sampling strategy.

    Here you might log the rejected sample, adjust the random seed, or modify the sampling
    grid based on which regions are rejected. For now, it simply prints the sample.
    """
    print("Rejected sample:", mapped_values)


def generate_concrete_parameter_samples(num_samples, numerical_parameters):
    # Step 1: Generate initial candidate points using adaptive LHS
    normalized_samples = adaptive_lhs_sampler(num_samples, numerical_parameters)

    concrete_samples = []
    for sample in normalized_samples:
        # Step 2: Map the normalized sample to concrete parameter values
        mapped_values = parameter_mapper(sample, numerical_parameters)

        # Step 3: Check if the mapped values satisfy all constraints
        if constraint_checker(mapped_values):
            concrete_samples.append(mapped_values)
        else:
            # Optionally log the rejected sample and provide feedback for adapting the LHS
            adjust_sampling_strategy(mapped_values)

    # Optionally run additional iterations to improve the sample set
    return concrete_samples


"""
def generate_concrete_parameter_samples(num_samples, parameter_specs):
    # Step 1: Generate initial candidate points using adaptive LHS
    normalized_samples = adaptive_lhs_sampler(num_samples, parameter_specs)

    concrete_samples = []
    for sample in normalized_samples:
        # Step 2: Map the normalized sample to concrete parameter values
        mapped_values = parameter_mapper(sample, parameter_specs)

        # Step 3: Check if the mapped values satisfy all the constraints
        if constraint_checker(mapped_values):
            concrete_samples.append(mapped_values)
        else:
            # Optionally log the rejected sample and provide feedback
            adjust_sampling_strategy(mapped_values)

    # Optionally run additional iterations to improve the sample set
    return concrete_samples
"""