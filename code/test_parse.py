import argparse
import yaml
import os
import sys
from typing import List, Tuple, Dict, Any


def get_args(default_argv):
    parser = argparse.ArgumentParser()
    for key, value in default_argv.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    return args


def check_arg_changes(argv: list[str], schema: Dict[str, Any]) -> list[str]:
    """Detect which keys were specified on the command line."""

    if len(argv) <= 1:
        return []

    tail = argv[1:]
    if len(tail) % 2 != 0:
        raise ValueError("Expected --key value pairs; got an odd number of tokens.")

    overridden: List[str] = []
    for key_tok, val_tok in zip(tail[0::2], tail[1::2]):
        if not key_tok.startswith("--"):
            raise ValueError(f"Keys must start with '--', got: {key_tok!r}")
        k = key_tok[2:]
        if k not in schema:
            raise KeyError(f"Unknown option '--{k}'. Allowed keys: {', '.join(sorted(schema.keys()))}")
        expected_type = str if schema[k] is None else type(schema[k])
        if expected_type in (int, float):
            try:
                _ = expected_type(val_tok)
            except ValueError:
                raise TypeError(f"Value for --{k} must be {expected_type.__name__}, got {val_tok!r}")
        overridden.append(k)

    return overridden


if __name__ == "__main__":
    with open("default_config.yaml", "r") as f:
        defaults = yaml.safe_load(f)
    check_arg_changes(sys.argv, defaults)
    args = get_args(defaults)
    print("Final arguments:", args)
