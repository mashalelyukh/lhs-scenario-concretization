import re  # Core Functionality: manipulating strings based on patterns
#to think about !=
from lark import Lark

def extract_parameters(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    numerical_parameters = {}
    enum_parameters = {}

    # Pattern 1: param in (min, max)
    pattern_range_constructor = r"(?P<param>\w+)\s*in\s*range\(\s*(?P<min>-?\d+(?:\.\d+)?)\s*,\s*(?P<max>-?\d+(?:\.\d+)?)\s*\)"
    for match in re.finditer(pattern_range_constructor, content):
        param = match.group("param")
        min_val = float(match.group("min"))
        max_val = float(match.group("max"))
        numerical_parameters[param] = {
            "min": min_val,
            "max": max_val,
            "source": "range_constructor"
        }

    # Pattern 1b: param in range(start, stop, step)
    pattern_range_func = r"(?P<param>\w+)\s*in\s*range\(\s*(?P<min>[\d.]+)\s*,\s*(?P<max>[\d.]+)\s*,\s*(?P<step>[\d.]+)\s*\)"
    for match in re.finditer(pattern_range_func, content):
        param = match.group("param")
        min_val = float(match.group("min"))
        max_val = float(match.group("max"))
        step = float(match.group("step"))
        numerical_parameters[param] = {
            "min": min_val,
            "max": max_val,
            "step": step,
            "source": "range"
        }

    # Pattern 1c: param in [minUnit..maxUnit] — DSL-style range with optional units
    pattern_range_dsl = r"""
    (?P<param>\w+)\s*in\s*\[\s*
    (?P<min>-?\d+(?:\.\d+)?)(?P<unit>\w+)?\s*
    \.\.\s*
    (?P<max>-?\d+(?:\.\d+)?)(?P=unit)?\s*
    \]
    """
    for match in re.finditer(pattern_range_dsl, content, re.VERBOSE):
        param = match.group("param")
        min_val = float(match.group("min"))
        max_val = float(match.group("max"))
        unit = match.group("unit") or ""

        numerical_parameters[param] = {
            "min": min_val,
            "max": max_val,
            "unit": unit,
            "source": "dsl_range"
        }

    # Pattern 2: constraints from keep(...) and cover(...) with various operators
    constraint_pattern = r"""
    (?P<type>keep|cover)              # Match 'keep' or 'cover'
    \(
    (?:default\s+)?                   # Optionally match 'default'
    it\.(?P<param>\w+)                # Parameter name
    \s*(?P<op>><=|=|==|<|>)           # Comparison operator; maybe do a single = here?
    \s*(?P<val>[\d.]+)                # Numeric value
    \s*(?P<unit>\w+)?                 # Optional unit
    \)
    """

    for match in re.finditer(constraint_pattern, content, re.VERBOSE):
        param = match.group("param")
        operator = match.group("op")
        value = float(match.group("val"))
        unit = match.group("unit") or ""

        # If the param isn't in the dict yet, initialize it
        if param not in numerical_parameters:
            numerical_parameters[param] = {
                "min": float('-inf'),
                "max": float('inf'),
                "unit": unit,
                "source": "constraint"
            }

        # Update min/max bounds based on the operator
        if operator in (">=", ">"):
            # Set minimum bound
            current_min = numerical_parameters[param]["min"]
            numerical_parameters[param]["min"] = max(current_min, value)
            numerical_parameters[param]["max"] = 130.0
            #numerical_parameters[param]["min"] + 50.0  # Placeholder upper bound
        elif operator in ("<=", "<"):
            # Set maximum bound
            current_max = numerical_parameters[param]["max"]
            numerical_parameters[param]["max"] = min(current_max, value)
            numerical_parameters[param]["min"] = 0.0
                    #numerical_parameters[param]["max"] - 50.0)  # Placeholder lower bound
        elif operator in ("=", "=="):
            # Fixed value: min and max are the same
            numerical_parameters[param]["min"] = value
            numerical_parameters[param]["max"] = value

    # Pattern 3: enum name: [value1, value2, ...]
    enum_pattern = r"enum\s+(?P<enum_name>\w+)\s*:\s*\[(?P<values>[^\]]+)\]"
    for match in re.finditer(enum_pattern, content):
        enum_name = match.group("enum_name")
        values_str = match.group("values")
        values = [v.strip() for v in values_str.split(",")]
        enum_parameters[enum_name] = values

    return numerical_parameters, enum_parameters
