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
    file_path = get_file_path() # asking for logical scenario input file

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

    #for testing:
    print("\n Concrete Parameter Samples using Adaptive LHS:")
    for i, sample in enumerate(concrete_samples, start=1):
        print(f"  Sample {i}: {sample}")

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

    param_bounds = {}
    for name, info_list in numerical_parameters.items():
        for idx, info in enumerate(info_list):
            flat_name = f"{name}_{idx}"
            param_bounds[flat_name] = (info["min"], info["max"])

    for name, info_list in enum_parameters.items():
        for idx, info in enumerate(info_list):
            flat_name = f"{name}_{idx}"
            param_bounds[flat_name] = info["values"]

    param_names = list(param_bounds.keys())
    X = [[sc[n] for n in param_names] for sc in concrete_samples]
    #X = [[sc["params"][n] for n in param_names] for sc in concrete_samples]
    y = [sc["criticality"] for sc in concrete_samples]

    bo = BayesianOptimizer(param_bounds)
    bo.fit(X, y)

    while True:
        # propose K new points
        K = get_int("How many new scenarios do you want to generate?", 10)
        candidates, preds = bo.propose(K, n_candidates=200)


        # generating new scenarios:
        existing_files = sorted(os.listdir(output_dir))
        N = len(existing_files)

        base_name, ext = os.path.splitext(os.path.basename(file_path))

        tmp_dir = os.path.join(output_dir, "._tmp_new")

        for offset, (x_vec, y_pred) in enumerate(zip(candidates, preds), start=1):
            # old: sample = {name: x_vec[j] for j, name in enumerate(param_names)}
            sample = correct_types_afterBO(
                x_vec,
                param_names,
                numerical_parameters,
                enum_parameters
            )

            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir)

            concretize_scenario(  # generating one scenario
                file_path,
                tmp_dir,
                [sample],
                flat_parameters
            )

            files = os.listdir(tmp_dir)  # exactly one
            if len(files) != 1:
                raise RuntimeError(f"Expected exactly one file in {tmp_dir}, found {files}")
            tmp_file = files[0]

            new_index = N + offset  # moving and renaming
            new_name = f"{base_name}_{new_index}{ext}"
            shutil.move(
                os.path.join(tmp_dir, tmp_file),
                os.path.join(output_dir, new_name)
            )

            print(f"for {new_name} the expected criticality value is {y_pred:.3f}")
        shutil.rmtree(tmp_dir, ignore_errors=True)


        if not ask_yes_no("\nWould you like to label these scenarios for criticality?"):
            print("Done here. Exiting BO loop.")
            break

        # run your real function on each
        #true_vals = []
        true_vals = get_labels(K)

        # augment & re‐fit
        bo.fit(bo.X_train + candidates,
               bo.y_train + true_vals)



"""
        # attach labels to the scenario list
        for scenario, crit in zip(concrete_samples, true_vals):
            scenario["criticality"] = crit

        print("\n Criticality labels saved:")
        for i, scen in enumerate(concrete_samples, 1):
            param_str = ", ".join(f"{n}={scen[n]}" for n in scen if n != "criticality")
            print(f"  Sample {i}: {{ {param_str} }} → criticality = {scen['criticality']:.3f}")
            # print(f"  Sample {i}: {scen['params']} → criticality = {scen['criticality']}")




        for x in candidates:
            y_true = evaluate_criticality(x)  # your real function
            params_str = ", ".join(f"{name}={val!r}"
                                   for (name, _, _), val in zip(opt.dims, x))
            print(f"Generated {params_str} → true criticality = {y_true:.3f}")
            true_vals.append(y_true)

        # augment & re‐fit
        bo.fit(bo.X_train + candidates,
                bo.y_train + true_vals)
"""

if __name__ == "__main__":
    main()
