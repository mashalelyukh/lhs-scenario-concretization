import os
import shutil
from lark_parser import extract_parameters
from lhs_sampler import generate_concrete_parameter_samples, flatten_parameters
from ranges_concretizer import concretize_scenario
from bayes_optimization import BayesianOptimizer
from mock_functions import f2, f3
from cli import (get_file_path, ask_num_samples, show_parameters, show_samples, ask_acquisition_function,
                 ask_new_scenario_count, show_generation_results, confirm_and_get_labels, show_label_summary)
from utils import correct_types_afterBO, encode_sample
from visualization import plot_parameter_ranges, plot_function_response
from file_manager import (init_output_dir, get_tmp_dir, ensure_clean_dir, count_scenarios, move_scenario_from_tmp,
                          refresh_output_dir)

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = init_output_dir(base_dir)

    file_path = get_file_path()  # asking for logical scenario input file

    numerical_parameters, enum_parameters = extract_parameters(file_path)
    show_parameters(numerical_parameters, enum_parameters)

    num_samples = ask_num_samples(default=10)

    concrete_samples = generate_concrete_parameter_samples(
        num_samples, numerical_parameters, enum_parameters
    )

    plot_parameter_ranges(numerical_parameters=numerical_parameters, enum_parameters=enum_parameters,
                          concrete_samples=concrete_samples)

    # for testing:
    show_samples(concrete_samples)

    """TO MOCK CRITICALITY FUNCTION CHOOSE FROM FOLLOWING PRINTS (OR COMMENT THEM ALL OUT)
        F2 IS CAPABLE OF TAKING ANY NUMBER OF PARAMETERS
        F3 IS A POLYNOM OF 5TH GRADE THAT TAKES ONLY ONE PARAMETER"""
    print(" ".join([str(f2(encode_sample(sample, numerical_parameters, enum_parameters))) for sample in concrete_samples]))
    #print(" ".join([str(f3(encode_sample(sample, numerical_parameters, enum_parameters))) for sample in concrete_samples]))

    flat_parameters = flatten_parameters(numerical_parameters, enum_parameters)

    concretize_scenario(file_path, output_dir, concrete_samples, flat_parameters)
    print(f"\nGenerated {num_samples} concrete scenarios in '{output_dir}'")

    refresh_output_dir(output_dir)

    # initial labeling (get the tuple of N floats in [0,1])
    initial_labels = confirm_and_get_labels(num_samples)
    if initial_labels is None:
        print("Exiting since no criticality labels provided.")
        return

    # attach labels to the scenario list
    for scenario, crit in zip(concrete_samples, initial_labels):
        scenario["criticality"] = crit

    show_label_summary(concrete_samples)

    # prepare for initial bayesian optimization
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
    y = [sc["criticality"] for sc in concrete_samples]

    acq = ask_acquisition_function()

    bo = BayesianOptimizer(param_bounds, acq_func=acq)
    bo.fit(X, y)

    loop_num = 1
    _, ext = os.path.splitext(os.path.basename(file_path))

    last_candidates = []

    # bayesian optimization loop (starts from second application of bo)
    while True:
        # propose K new points
        K = ask_new_scenario_count(default=10)
        candidates, preds = bo.propose(K, n_candidates=200)

        """TO MOCK CRITICALITY FUNCTION CHOOSE FROM FOLLOWING PRINTS (OR COMMENT THEM ALL OUT)
        F2 IS CAPABLE OF TAKING ANY NUMBER OF PARAMETERS
        F3 IS A POLYNOM OF 5TH GRADE THAT TAKES ONLY ONE PARAMETER"""

        print("Predicted criticalities (mocked with function on encoded samples):")
        formatted = []
        for x_vec in candidates:
            sample_dict = correct_types_afterBO(x_vec, param_names,
                                                numerical_parameters,
                                                enum_parameters)
            x_encoded = encode_sample(sample_dict,
                                      numerical_parameters,
                                      enum_parameters)
            formatted.append(str(f2(x_encoded)))
            #formatted.append(str(f3(x_encoded)))
        print(" ".join(formatted))

        # count existing .osc files
        N_existing = count_scenarios(output_dir, ext)

        new_samples = list()
        tmp_dir = get_tmp_dir(output_dir)
        for offset, (x_vec, y_pred) in enumerate(zip(candidates, preds), start=1):
            # convert types back to int/enum/float
            sample = correct_types_afterBO(x_vec, param_names, numerical_parameters, enum_parameters)
            new_samples.append(sample)
            # clean the temporary directory
            ensure_clean_dir(tmp_dir)

            # generating one scenario in tmp_dir
            concretize_scenario(file_path, tmp_dir, [sample], flat_parameters)

            new_name = move_scenario_from_tmp(
                tmp_dir, output_dir, N_existing + offset, loop_num, ext)
            show_generation_results([new_name], [y_pred])

        shutil.rmtree(tmp_dir, ignore_errors=True)

        # ask to label new scenarios
        loop_labels = confirm_and_get_labels(K)
        if loop_labels is None:
            print("Done here. Exiting BO loop.")
            last_candidates = candidates
            break

        for scenario, crit in zip(new_samples, loop_labels):
            scenario["criticality"] = crit

        show_label_summary(new_samples)

        """TO PLOT CHOOSE FROM FOLLOWING PRINTS ONE OF THE AVAILABLE MOCK FUNCTIONS
        (OR COMMENT BOTH LINES OUT)
        F2 IS CAPABLE OF TAKING ANY NUMBER OF PARAMETERS
        F3 IS A POLYNOM OF 5TH GRADE THAT TAKES ONLY ONE PARAMETER"""
        plot_function_response(f2, numerical_parameters, enum_parameters=enum_parameters, bo=bo, concrete_samples=concrete_samples, x_sel=candidates)
        #plot_function_response(f3, numerical_parameters, enum_parameters=enum_parameters, bo=bo, concrete_samples=concrete_samples, x_sel=candidates)

        # re‐fit
        concrete_samples.extend(new_samples)
        bo.fit(bo.X_train + candidates, bo.y_train + loop_labels)
        loop_num += 1

    """TO PLOT CHOOSE FROM FOLLOWING PRINTS ONE OF THE AVAILABLE MOCK FUNCTIONS
        (OR COMMENT BOTH LINES OUT )
        F2 IS CAPABLE OF TAKING ANY NUMBER OF PARAMETERS
        F3 IS A POLYNOM OF 5TH GRADE THAT TAKES ONLY ONE PARAMETER"""
    plot_function_response(f2, numerical_parameters, enum_parameters, bo, concrete_samples, last_candidates)
    #plot_function_response(f3, numerical_parameters, enum_parameters, bo, concrete_samples, last_candidates)


if __name__ == "__main__":
    main()
