import os
import argparse
from typing import Any, Dict, List

import torch
import yaml
from inspect import currentframe, getframeinfo
from utils.mydevice import mydevice


accept_changes = ('model_id', 'pwnet_layer', 'pwnet_dim', 'trans_layer', 'trans_dim', 'window_sliding',
                  'maxlr', 'batch_size', 'b', 'poly_deg', 'dpt_train', 'dpt_valid', 'load_weight', 'end_epoch')

# ===================================================
def assert_nan(x):
    cframe = currentframe().f_back
    filename = getframeinfo(cframe).filename
    lineno = cframe.f_lineno
    masknan = torch.isnan(x)
    if masknan.any() == True:
        print(filename, ' line ', lineno, ' has nan')
        quit()


# ===================================================
def print_compute_tree(name,node):
    dot = make_dot(node)
    #print(dot)
    dot.render(name)


# ===================================================
def check_data(loader,data_set,tau_traj_len,tau_long,tau_short,nitr,append_strike):

    label_idx = int(tau_traj_len//tau_long)
    for qpl_input,qpl_label in loader.train_loader:

        q_traj,p_traj,q_label,p_label,l_init = pack_data(qpl_input,qpl_label)
        data_set.check_md_trajectory(q_traj,p_traj,q_label,p_label,l_init,label_idx,
                                     tau_short,nitr,append_strike)


# ===================================================

def pack_data(qpl_input, qpl_label):

    q_traj = qpl_input[:,0,:,:,:].clone().detach() #.requires_grad_()
    q_traj = q_traj.permute(1,0,2,3)
    # shape [trajectory,nsamples,nparticles,dim]
    p_traj = qpl_input[:,1,:,:,:].clone().detach() #.requires_grad_()
    p_traj = p_traj.permute(1,0,2,3)
    l_init = qpl_input[:,2,0,:,:].clone().detach() #.requires_grad_()
    # l_init.shape is [nsamples,nparticles,DIM]
    q_label = qpl_label[:,0,:,:,:].clone().detach() #.requires_grad_()
    p_label = qpl_label[:,1,:,:,:].clone().detach() #.requires_grad_()

    q_traj = mydevice.load(q_traj)
    p_traj = mydevice.load(p_traj)
    l_init = mydevice.load(l_init)
    q_label = mydevice.load(q_label)
    p_label = mydevice.load(p_label)

    return q_traj,p_traj,q_label,p_label,l_init


# ===================================================

def print_dict(name, this_dict, log_file=None):
    print(name, 'dict ============== ')
    for key, value in this_dict.items():
        print(key, ':', value)
    # Write to log file if provided
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(f"{name} dict ============== \n")
            for key, value in this_dict.items():
                f.write(f"{key} : {value}\n")
            f.write('\n')


def check_arg_changes(argv: list[str], schema: Dict[str, Any]) -> list[str]:
    """Detect which keys were specified on the command line."""

    if len(argv) <= 1:
        return []

    tail = argv[1:]
    print('tail of argv:', tail)
    if len(tail) % 2 != 0:
        raise ValueError("Expected --key value pairs; got an odd number of tokens.")

    overridden: List[str] = []
    for key_tok, val_tok in zip(tail[0::2], tail[1::2]):
        if not key_tok.startswith("--"):
            raise ValueError(f"Keys must start with '--', got: {key_tok!r}")
        k = key_tok[2:]
        if k not in schema:
            raise KeyError(f"Unknown option '--{k}'. Existing keys: {', '.join(sorted(schema.keys()))}")
        if k not in accept_changes:
            raise KeyError(f"Unacceptable option '--{k}'. Acceptable keys: {', '.join(sorted(accept_changes))}")
        expected_type = str if schema[k] is None else type(schema[k])
        if expected_type in (int, float):
            try:
                _ = expected_type(val_tok)
            except ValueError:
                raise TypeError(f"Value for --{k} must be {expected_type.__name__}, got {val_tok!r}")
        overridden.append(k)

    return overridden


def get_args(default_argv):
    parser = argparse.ArgumentParser()
    for key, value in default_argv.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    return args
