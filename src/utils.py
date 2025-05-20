import os
import shutil


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


def clear_output_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # remove file or symbolic link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # remove subdirectory
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


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


def correct_types_afterBO(x_vec, param_names, numerical_parameters, enum_parameters):
    sample = {}
    for j, flat_name in enumerate(param_names):
        raw = x_vec[j]
        base, idx_str = flat_name.rsplit("_", 1)
        idx = int(idx_str)

        # Numeric params
        if base in numerical_parameters:
            info = numerical_parameters[base][idx]
            t = info["type_annotation"]
            # unit = info.get("unit", "")
            if t in ("int", "uint"):
                val = int(round(raw))
            else:
                val = float(raw)
            # sample[flat_name] = f"{val}{unit}" if unit else val
            sample[flat_name] = val

        # Enum params
        elif base in enum_parameters:
            info = enum_parameters[base][idx]
            cats = info["values"]
            if isinstance(raw, str):
                sample[flat_name] = raw
            else:
                sample[flat_name] = cats[int(round(raw))]

        else:
            raise KeyError(f"Unknown parameter base '{base}'")

    return sample


def encode_enum(sample: dict, enum_parameters: dict) -> list[float]:
    numeric = []
    for full_key, val in sample.items():
        # full_key is e.g. "speed_0" or "weatherCondition_0"
        base = full_key.rsplit("_", 1)[0]
        if base in enum_parameters:
            # look up the list of possible strings
            enum_vals = enum_parameters[base][0]["values"]
            # map "rainy"->2 etc (index+1)
            numeric.append(enum_vals.index(val) + 1)
        else:
            # it's already numeric
            numeric.append(val)
    return numeric


def encode_sample(sample: dict,
                  numerical_parameters: dict,
                  enum_parameters: dict) -> list[float]:
    """
    Turn one concrete-sample dict into a pure-numeric list,
    first all numeric params (in parser order), then all enums
    as 1,2,3… in the order they were declared.
    """
    x = []
    # 1) numeric first, in the same order extract_parameters gave you:
    for name, infos in numerical_parameters.items():
        for idx, info in enumerate(infos):
            key = f"{name}_{idx}"
            x.append(sample[key])

    # 2) now encode each enum:
    for name, infos in enum_parameters.items():
        # there may be multiple enum‐slots, but typically idx=0
        for idx, info in enumerate(infos):
            key = f"{name}_{idx}"
            raw = sample[key]  # e.g. "foggy"
            values = info["values"]  # e.g. ["sunny","rainy","foggy",…]
            x.append(values.index(raw) + 1)  # 1-based encoding

    return x
