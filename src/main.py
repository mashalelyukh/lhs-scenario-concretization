import os
import shutil
from lark_parser import extract_parameters
#from parser import extract_parameters
from lhs_sampler import generate_concrete_parameter_samples, flatten_parameters
from ranges_concretizer import concretize_scenario
#from bayesian_optimization import BayesianOptimizer
from bayes_optimization2 import BayesianOptimizer
from utils import ask_yes_no, get_file_path, clear_output_folder, get_labels, get_int, get_float, correct_types_afterBO

def main():
    file_path = get_file_path() # asking for logical scenario file

    numerical_parameters, enum_parameters = extract_parameters(file_path)
    print("\n Numerical Parameters:")
    for name, info in numerical_parameters.items():
        print(f"  • {name}: {info}")
    print(" Enum Parameters:")
    for enum_name, values in enum_parameters.items():
        print(f"  • {enum_name}: {values}")

    default_N = 10
    #input_N = input(f"\nEnter number of concrete scenarios (an integer) to generate [default {default_N}]: ").strip()
    num_samples = get_int(
        "Enter number of concrete scenarios (an integer) to generate: ",
        default_N
    )

    base_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(base_dir, "scenarios_output")
    clear_output_folder(output_dir)

    concrete_samples = generate_concrete_parameter_samples(
        num_samples, numerical_parameters, enum_parameters
    )

    flat_parameters = flatten_parameters(numerical_parameters, enum_parameters)

    concretize_scenario(file_path, output_dir, concrete_samples, flat_parameters)
    print(f"\nGenerated {num_samples} concrete scenarios in '{output_dir}'")

    # whether user wants to label the scenarios
    if not ask_yes_no("\nWould you like to label these scenarios for criticality?"):
        print("Exiting since no criticality labels provided.")
        return

    # get the tuple of N floats in [0,1]
    labels = get_labels(num_samples)

    # attach labels to the scenario list
    for scenario, crit in zip(concrete_samples, labels):
        scenario["criticality"] = crit

    print("\n Criticality labels saved:")
    for i, scen in enumerate(concrete_samples, 1):
        param_str = ", ".join(f"{n}={scen[n]}" for n in scen if n != "criticality")
        print(f"  Sample {i}: {{ {param_str} }} → criticality = {scen['criticality']:.3f}")
        #print(f"  Sample {i}: {scen['params']} → criticality = {scen['criticality']}")







# 1) Build param_bounds from what you extracted earlier
    #    (same logic you used to feed your LHS sampler)
    param_bounds = {}
    for name, info_list in numerical_parameters.items():
        for idx, info in enumerate(info_list):
            flat_name = f"{name}_{idx}"
            param_bounds[flat_name] = (info["min"], info["max"])

        #param_bounds[name] = (info["min"], info["max"])
    #for name, cats in enum_parameters.items():
        #param_bounds[name] = cats  # list → treated as categorical

    for name, info_list in enum_parameters.items():
        for idx, info in enumerate(info_list):
            flat_name = f"{name}_{idx}"
            param_bounds[flat_name] = info["values"]

    # 2) Prepare X, y for BO
    param_names = list(param_bounds.keys())
    X = [[sc[n] for n in param_names] for sc in concrete_samples]
    #X = [[sc["params"][n] for n in param_names] for sc in concrete_samples]
    y = [sc["criticality"] for sc in concrete_samples]

    # 3) Instantiate & fit
    bo = BayesianOptimizer(param_bounds)
    bo.fit(X, y)

    # 4) Ask user how many new scenarios and the desired threshold
    K = get_int("How many new scenarios do you want to generate?", 10)
    C = get_float("Enter the desired minimum criticality (a float betwen 0 and 1):", 0.0, 1.0)

    # 5) Finally, propose with BO
    #new_X, new_y = bo.propose(K, C)
    new_X, new_y = bo.propose(K, C, 100)

    # 6) Hand `new_X` into your concretization pipeline …
    #    (convert each x back into a dict of params, then call concretize_scenario)
    # ────────────────────────────────────────────────────────────────────────────




    # 6) Append K new scenarios by generating into a temp folder and moving them
    # Count how many files already exist so new ones start at N+1
    existing_files = sorted(os.listdir(output_dir))
    N = len(existing_files)

    # Name‐parts for renaming
    base_name, ext = os.path.splitext(os.path.basename(file_path))

    # Temporary directory to catch one file at a time
    tmp_dir = os.path.join(output_dir, "._tmp_new")

    for offset, (x_vec, y_pred) in enumerate(zip(new_X, new_y), start=1):
        # 1) Build the single-sample dict
        #old: sample = {name: x_vec[j] for j, name in enumerate(param_names)}
        sample = correct_types_afterBO(
            x_vec,
            param_names,
            numerical_parameters,
            enum_parameters
        )
        # 2) Prepare a fresh temp folder
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)

        # 3) Generate into the temp folder
        concretize_scenario(
            file_path,
            tmp_dir,  # ← generate here
            [sample],
            flat_parameters
        )

        # 4) There must now be exactly one file in tmp_dir
        files = os.listdir(tmp_dir)
        if len(files) != 1:
            raise RuntimeError(f"Expected exactly one file in {tmp_dir}, found {files}")
        tmp_file = files[0]

        # 5) Move & rename it into output_dir
        new_index = N + offset
        new_name = f"{base_name}_{new_index}{ext}"
        shutil.move(
            os.path.join(tmp_dir, tmp_file),
            os.path.join(output_dir, new_name)
        )

        # 6) Print feedback
        print(f"for {new_name} the expected criticality value is {y_pred:.3f}")

    # 7) Clean up the temp folder
    shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
