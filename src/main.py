# src/main.py
import os
from lark_parser import extract_parameters
#from parser import extract_parameters

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

if __name__ == "__main__":
    main()
