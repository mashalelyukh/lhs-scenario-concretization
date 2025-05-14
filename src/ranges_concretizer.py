import os
import copy


def replace_range_with_value(line, span, concrete_value, param_spec):
    # formatting the concrete value
    if param_spec.get('enum'):
        value_str = f'"{concrete_value}"'
    else:
        value_str = f"{concrete_value}{param_spec['unit']}" if param_spec.get('unit') else str(concrete_value)

    # wrap into correct replacement expression
    param_name = param_spec['param']
    param_type = param_spec['type']
    extras = param_spec.get('extras', '').strip()

    if param_type.startswith('enum') or param_type == 'in':
        replacement = f"{param_name} == {value_str}"
    elif param_type == 'colon':
        replacement = f"{param_name}: {value_str}"
    elif param_type == 'call':
        if extras:
            replacement = f"{param_name}({value_str}, {extras})"
        else:
            replacement = f"{param_name}({value_str})"
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")

    # replacing the text at the given span:
    start, end = span
    new_line = line[:start] + replacement + line[end:]
    return new_line

def concretize_scenario(file_path, output_dir, samples, flat_parameters):
    with open(file_path, 'r') as f:
        original_lines = f.readlines()

    os.makedirs(output_dir, exist_ok=True)

    for sample_idx, sample in enumerate(samples, start=1):
        # Work on a copy of the original lines for this sample
        scenario_lines = copy.deepcopy(original_lines)

        # Sort parameter occurrences by line number (optional but helps clarity)
        param_occurrences = []
        for param_key, concrete_value in sample.items():
            param_name = param_key.rsplit('_', 1)[0]  # Remove _0 index suffix
            specs = flat_parameters[param_key]
            param_occurrences.append((specs['line_number'], specs['span'], concrete_value, specs))

        # Process parameters sorted from bottom to top to not mess up spans
        param_occurrences.sort(key=lambda x: (x[0], -x[1][0]))

        for line_number, span, concrete_value, param_spec in param_occurrences:
            line_idx = line_number - 1  # Convert to 0-based index
            scenario_lines[line_idx] = replace_range_with_value(
                scenario_lines[line_idx],
                span,
                concrete_value,
                param_spec
            )

        output_file = os.path.join(output_dir, f"scenario_{sample_idx}.osc")
        with open(output_file, 'w') as f_out:
            f_out.writelines(scenario_lines)

        print(f"Generated: {output_file}")

