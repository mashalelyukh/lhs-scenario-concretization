# src/main.py
import os
from lark_parser import extract_parameters
#from parser import extract_parameters
from sampler import generate_concrete_parameter_samples

def main():
    # Dynamically find the project root (one level up from /src/)
    base_dir = os.path.dirname(os.path.dirname(__file__))

    # Join the path safely to get: lhs-scenario-concretization/scenarios_input/ex1.osc
    file_path = os.path.join(base_dir, "scenarios_input", "ex2.osc")

    #file_path = "scenarios_input/ex1.osc"  # Replace with your test file
    numerical_parameters, enum_parameters = extract_parameters(file_path)

    print("📦 Numerical Parameters:")
    for name, info in numerical_parameters.items():
        print(f"  {name}: {info}")

    print("\n🔤 Enum Parameters:")
    for enum_name, values in enum_parameters.items():
        print(f"  {enum_name}: {values}")

    num_samples = 10
    concrete_samples = generate_concrete_parameter_samples(num_samples, numerical_parameters)

    print("\n🎯 Concrete Parameter Samples using Adaptive LHS:")
    for i, sample in enumerate(concrete_samples, start=1):
        print(f"  Sample {i}: {sample}")

if __name__ == "__main__":
    main()
