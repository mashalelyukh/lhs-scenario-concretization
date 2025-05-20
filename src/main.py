import os
import shutil
import time
import glob
from lark_parser import extract_parameters
# from parser import extract_parameters
from lhs_sampler import generate_concrete_parameter_samples, flatten_parameters
from ranges_concretizer import concretize_scenario
# from bayesian_optimization import BayesianOptimizer
from bayes_optimization2 import BayesianOptimizer
from testing_functions import f2, f3
from utils import ask_yes_no, get_file_path, clear_output_folder, get_labels, get_int, get_float, correct_types_afterBO, encode_sample
from visualisation import plot_parameter_ranges, plot_function_response

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(base_dir, "scenarios_output")
    # clear_output_folder(output_dir)
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


    file_path = get_file_path()  # asking for logical scenario input file

    numerical_parameters, enum_parameters = extract_parameters(file_path)
    print("\n Numerical Parameters:")
    for name, info in numerical_parameters.items():
        print(f"  • {name}: {info}")
    print(" Enum Parameters:")
    for enum_name, values in enum_parameters.items():
        print(f"  • {enum_name}: {values}")

    default_N = 10
    # input_N = input(f"\nEnter number of concrete scenarios (an integer) to generate [default {default_N}]: ").strip()
    num_samples = get_int(
        "Enter number of concrete scenarios (an integer) to generate: ",
        default_N
    )

    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(base_dir, "scenarios_output")
    #clear_output_folder(output_dir)
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    """

    concrete_samples = generate_concrete_parameter_samples(
        num_samples, numerical_parameters, enum_parameters
    )

    plot_parameter_ranges(numerical_parameters=numerical_parameters,enum_parameters=enum_parameters,
                          concrete_samples=concrete_samples)

    # for testing:
    print("\n Concrete Parameter Samples using Adaptive LHS:")
    for i, sample in enumerate(concrete_samples, start=1):
        print(f"  Sample {i}: {sample}")


    """TO MOCK CRITICALITY FUNCTION CHOOSE FROM FOLLOWING PRINTS (OR COMMENT THEM ALL OUT)"""
    #print(" ".join([str(f2(encode_sample(sample, numerical_parameters, enum_parameters))) for sample in concrete_samples]))
    print(" ".join([str(f3(encode_sample(sample, numerical_parameters, enum_parameters))) for sample in concrete_samples]))

    flat_parameters = flatten_parameters(numerical_parameters, enum_parameters)

    concretize_scenario(file_path, output_dir, concrete_samples, flat_parameters)
    print(f"\nGenerated {num_samples} concrete scenarios in '{output_dir}'")

    parent = os.path.dirname(output_dir)
    temp = os.path.join(parent, "._so_tmp")
    try:
        os.rename(output_dir, temp)
        os.rename(temp, output_dir)
    except Exception as e:
        pass


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
        # print(f"  Sample {i}: {scen['params']} → criticality = {scen['criticality']}")

    # fir initial bayesian optimization
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
    # X = [[sc["params"][n] for n in param_names] for sc in concrete_samples]
    y = [sc["criticality"] for sc in concrete_samples]

    # — let the user pick UCB, EI or PI —
    valid = {"UCB", "EI", "PI"}
    while True:
        acq = input(
            'Which acquisition function would you like to use for BO? (type "UCB", "EI", or "PI"): ').strip().upper()
        if acq in valid:
            break
        print(f'  "{acq}" is not one of {valid}. Please try again.')

    bo = BayesianOptimizer(param_bounds, acq_func=acq)
    bo.fit(X, y)

    loop_num = 1
    _, ext = os.path.splitext(os.path.basename(file_path))

    last_candidates = []
    last_preds = None

    # bayesian optimization loop (starts from second application of bo)
    while True:
        # propose K new points
        K = get_int("How many new scenarios do you want to generate?", 10)
        candidates, preds = bo.propose(K, n_candidates=200)

        """TO MOCK CRITICALITY FUNCTION CHOOSE FROM FOLLOWING PRINTS (OR COMMENT THEM ALL OUT)"""

        print("Predicted criticalities (mocked with function on encoded samples):")
        formatted = []
        for x_vec in candidates:
            sample_dict = correct_types_afterBO(x_vec, param_names,
                                                numerical_parameters,
                                                enum_parameters)
            x_encoded = encode_sample(sample_dict,
                                      numerical_parameters,
                                      enum_parameters)
            #formatted.append(str(f2(x_encoded)))
            formatted.append(str(f3(x_encoded)))
        print(" ".join(formatted))

        # count existing .osc files
        existing = [
            f for f in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, f))
               and f.endswith(ext)
               and f.startswith("scenario_")
        ]
        N = len(existing)

        # generating new scenarios:
        # existing_files = sorted(os.listdir(output_dir))
        # N = len(existing_files)

        # base_name, ext = os.path.splitext(os.path.basename(file_path))

        tmp_dir = os.path.join(output_dir, "._tmp_new")
        new_samples = list()
        for offset, (x_vec, y_pred) in enumerate(zip(candidates, preds), start=1):
            # convert types back to int/enum/float
            sample = correct_types_afterBO(
                x_vec,
                param_names,
                numerical_parameters,
                enum_parameters
            )
            new_samples.append(sample)
            # clean and make temp-directory
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir)

            # generating one scenario in tmp_dir
            concretize_scenario(
                file_path,
                tmp_dir,
                [sample],
                flat_parameters
            )

            files = os.listdir(tmp_dir)  # exactly one
            if len(files) != 1:
                raise RuntimeError(f"Expected exactly one file in {tmp_dir}, found {files}")
            tmp_file = files[0]

            new_index = N + offset  # existing count + offset
            new_name = f"scenario_{new_index}_{loop_num}BO_loop{ext}"
            shutil.move(
                os.path.join(tmp_dir, tmp_file),
                os.path.join(output_dir, new_name)
            )

            print(f"for {new_name} the expected criticality value is {y_pred:.3f}")

            parent = os.path.dirname(output_dir)
            temp = os.path.join(parent, "._so_tmp")
            try:
                os.rename(output_dir, temp)
                os.rename(temp, output_dir)
            except Exception as e:
                pass

            #os.utime(refresh_marker, None)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        if not ask_yes_no("\nWould you like to label these scenarios for criticality?"):
            print("Done here. Exiting BO loop.")
            last_candidates = candidates
            last_preds = preds
            break

        true_vals = get_labels(K)

        for scenario, crit in zip(new_samples, true_vals):
            scenario["criticality"] = crit

        print("\n Criticality labels saved:")

        for i, scen in enumerate(new_samples, 1):
            param_str = ", ".join(f"{n}={scen[n]}" for n in scen if n != "criticality")
            print(f"  Sample {i}: {{ {param_str} }} → criticality = {scen['criticality']:.3f}")


        """TO PLOT A FUNCTION CHOOSE FROM FOLLOWING PRINTS (OR COMMENT THEM ALL OUT)"""
        #plot_function_response(f2, numerical_parameters, enum_parameters=enum_parameters, bo=bo, concrete_samples=concrete_samples, x_sel=candidates)
        plot_function_response(f3, numerical_parameters, enum_parameters=enum_parameters, bo=bo, concrete_samples=concrete_samples, x_sel=candidates)

        concrete_samples.extend(new_samples)

        # re‐fit
        bo.fit(bo.X_train + candidates,
               bo.y_train + true_vals)

        loop_num += 1

    """TO PLOT A FUNCTION CHOOSE FROM FOLLOWING PRINTS (OR COMMENT THEM ALL OUT)"""
    #plot_function_response(f2, numerical_parameters, enum_parameters, bo, concrete_samples, last_candidates)
    plot_function_response(f3, numerical_parameters, enum_parameters, bo, concrete_samples, last_candidates)


if __name__ == "__main__":
    main()
