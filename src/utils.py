
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


# turns one concrete-sample dict into a pure-numeric list
def encode_sample(sample: dict,
                  numerical_parameters: dict,
                  enum_parameters: dict) -> list[float]:
    x = []
    # numeric first, in the same order extract_parameters gave you:
    for name, infos in numerical_parameters.items():
        for idx, info in enumerate(infos):
            key = f"{name}_{idx}"
            x.append(sample[key])

    # now encode each enum:
    for name, infos in enum_parameters.items():
        # there may be multiple enum‐slots, but typically idx=0
        for idx, info in enumerate(infos):
            key = f"{name}_{idx}"
            raw = sample[key]  # e.g. "foggy"
            values = info["values"]  # e.g. ["sunny","rainy","foggy",…]
            x.append(values.index(raw) + 1)  # 1-based encoding

    return x


