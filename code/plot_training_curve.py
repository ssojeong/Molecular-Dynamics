import os, sys
import yaml
import subprocess

from utils import utils


def replace_xx_in_dict(replacement):
    """
    Replace all 'xx' in the template dict with a given string.
    Args:
        replacement (str): String to replace 'xx'
    """
    with open("template.dict", "r") as f:
        template = f.read()[1:]

    print(replacement)
    # Replace all occurrences of "xx"
    updated_content = template.replace("xx", replacement)

    try:
        with open("working.dict", "r") as f:
            existing = f.read()[:-1]
    except FileNotFoundError:
            existing = '{'

    print(type(updated_content), updated_content)
    print(type(existing), existing)
    with open("working.dict", "w") as f:
        f.write(existing+updated_content)


def one_plot(argv_list):
    overridden_argv = utils.check_arg_changes(argv_list, default_args)
    main_args = utils.get_args(default_args)

    ignore_list = ['model_id', 'batch_size', 'load_weight', 'end_epoch', 'poly_deg', 'window_sliding']
    overridden_argv = [k for k in overridden_argv if k not in ignore_list]
    if len(overridden_argv) == 0:
        main_model_name = 'vanilla'
    else:
        main_model_name = '-'.join([f'{k}={getattr(main_args, k)}' for k in sorted(overridden_argv)])
    # print(main_model_name)
    log_file_path = f"./logfile/{main_model_name}_{default_args['model_id']}.log"
    print(log_file_path)
    _ = subprocess.run(["./show_results.sh", log_file_path], capture_output=True, text=True)
    replace_xx_in_dict(log_file_path)

    cmd = ["python", "loss_weight_ws1.py", "working.dict", log_file_path[10:-4], "0", log_file_path[10:-4]]
    print("Running", cmd)
    # Run the command in background (like & in bash)
    process = subprocess.Popen(cmd)


if __name__ == '__main__':
    yaml_config_path = 'default_config.yaml'
    with open(yaml_config_path, 'r') as f:
        default_args = yaml.load(f, Loader=yaml.Loader)

    experiment_list = [
        'python maintrain.py --load_weight none'
    ]
    try:
        os.remove("working.dict")
    except FileNotFoundError:
        pass

    for exp in experiment_list:
        exp = exp.split()
        one_plot(exp[1:])
