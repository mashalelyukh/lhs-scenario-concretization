import os
import shutil


# (re-)initialize the output directory for scenarios.
# returns a full path to the created output directory
def init_output_dir(base_dir, dir_name="scenarios_output"):

    output_dir = os.path.join(base_dir, dir_name)
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# computes temporary directory inside the output directory for staging new scenarios.
def get_tmp_dir(output_dir, tmp_name="._tmp_new"):

    return os.path.join(output_dir, tmp_name)


# ensures given path is an empty directory
def ensure_clean_dir(path):

    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


# counts how many scenario files the output directory contains
def count_scenarios(output_dir, ext, prefix="scenario_"):

    return len([
        f for f in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, f))
           and f.startswith(prefix)
           and f.endswith(ext)
    ])


# used for moving a single scenario from tmp_dir into output_dir, and renaming it
def move_scenario_from_tmp(tmp_dir, output_dir, scenario_index, loop_num, ext, prefix="scenario"):

    files = os.listdir(tmp_dir)
    if len(files) != 1:
        raise RuntimeError(f"Expected exactly one file in {tmp_dir}, found {files}")
    src = os.path.join(tmp_dir, files[0])
    new_name = f"{prefix}_{scenario_index}_{loop_num}BO_loop{ext}"
    dst = os.path.join(output_dir, new_name)
    shutil.move(src, dst)
    refresh_output_dir(output_dir)
    return new_name


# forces a directory refresh
def refresh_output_dir(output_dir):

    parent = os.path.dirname(output_dir)
    temp = os.path.join(parent, "._so_tmp")
    try:
        os.rename(output_dir, temp)
        os.rename(temp, output_dir)
    except Exception:
        pass
