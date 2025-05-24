import os
import shutil
from utils import correct_types_afterBO, encode_sample


def get_file_path(prompt_msg="Please enter the path to your logical scenario (.osc) file: "):
    while True:
        path = input(prompt_msg).strip()
        if os.path.isfile(path):
            return path
        print("File not found. Please try again.")


def ask_yes_no(question):
    while True:
        ans = input(f"{question} (yes/no): ").strip().lower()
        if ans == "yes":
            return True
        if ans == "no":
            return False
        print("Please answer 'yes' or 'no'.")



def get_labels(n):
    example = " ".join(f"{(i + 1) / (n + 1):.3f}" for i in range(n))
    print(f"\nPlease enter {n} criticality values (floats between 0 and 1) separated by spaces.")
    print(f"Example for N={n}: {example}")
    while True:
        raw = input("→ ").strip()
        parts = raw.split()
        if len(parts) != n:
            print(f" You entered {len(parts)} values; expected {n}. Try again.")
            continue
        try:
            values = [float(p.replace(",", ".")) for p in parts]
        except ValueError:
            print("All entries must be numbers. Try again.")
            continue
        if any(v < 0 or v > 1 for v in values):
            print("All values must be between 0 and 1. Try again.")
            continue

        print(f"\nYou entered: {values}")
        if ask_yes_no("Confirm these values?"):
            return values
        print("Let's try again.")


def get_int(prompt, default=None):
    while True:
        raw = input(f"{prompt}{' [default ' + str(default) + ']' if default is not None else ''}: ").strip()
        if raw == "" and default is not None:
            return default
        if raw.isdigit():
            return int(raw)
        print("Please enter a valid integer.")


def get_float(prompt, min_val=None, max_val=None):
    while True:
        raw = input(f"{prompt}: ").strip().replace(",", ".")
        try:
            val = float(raw)
        except ValueError:
            print("Please enter a valid number.")
            continue
        if (min_val is not None and val < min_val) or (max_val is not None and val > max_val):
            rng = []
            if min_val is not None: rng.append(f"≥{min_val}")
            if max_val is not None: rng.append(f"≤{max_val}")
            print(f"Value must be {' and '.join(rng)}. Try again.")
            continue
        return val


# prints out the numerical and enum parameters
def show_parameters(
    numerical_parameters, enum_parameters):
    print("\nNumerical Parameters:")
    for name, infos in numerical_parameters.items():
        print(f"  • {name}: {infos}")
    print("Enum Parameters:")
    for name, values in enum_parameters.items():
        print(f"  • {name}: {values}")


def ask_num_samples(default: int = 10) -> int:
    return get_int(
        "Enter number of concrete scenarios (an integer) to generate: ",
        default
    )


def show_samples(samples):
    print("Concrete Parameter Samples using LHS")
    for i, sample in enumerate(samples, start=1):
        print(f"  Sample {i}: {sample}")


def ask_acquisition_function():
    valid = {"UCB", "EI", "PI"}
    while True:
        acq = input(
            'Which acquisition function would you like to use for BO? (type "UCB", "EI", or "PI"): ').strip().upper()
        if acq in valid:
            return acq
        print(f'  "{acq}" is not one of {valid}. Please try again.')


def ask_new_scenario_count(default: int = 10) -> int:
    return get_int(
        "How many new scenarios do you want to generate? ",
        default
    )


# prints criticalities, predicted by a mock function
def show_predictions(
    candidates,
    param_names,
    numerical_parameters,
    enum_parameters,
    transform_fn,
    description: str = "function on encoded samples"
):
    print(f"\nPredicted criticalities (mocked with {description}):")
    formatted = []
    for x_vec in candidates:
        sample_dict = correct_types_afterBO(
            x_vec,
            param_names,
            numerical_parameters,
            enum_parameters
        )
        x_encoded = encode_sample(
            sample_dict,
            numerical_parameters,
            enum_parameters
        )
        formatted.append(str(transform_fn(x_encoded)))
    print(" ".join(formatted))


# report file names and their predicted criticalities
def show_generation_results(
    new_names, preds):
    for name, pred in zip(new_names, preds):
        print(f"for {name} the expected criticality value is {pred:.3f}")


def confirm_and_get_labels(n: int):
    if not ask_yes_no("\nWould you like to label these scenarios for criticality?"):
        return None
    return get_labels(n)


# for each sample print parameters & criticality
def show_label_summary(samples):
    print("\nCriticality labels saved:")
    for i, scen in enumerate(samples, start=1):
        param_str = ", ".join(
            f"{k}={scen[k]}" for k in scen if k != "criticality"
        )
        print(f"  Sample {i}: {{ {param_str} }} → criticality = {scen['criticality']:.3f}")
