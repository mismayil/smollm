"""
Script that unifies several nanotron yaml config files to a single file.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import yaml

OVERRIDE_FORMAT_INFO = (
    "Each override string should be specified with the format `k1/k2/k3/.../kn:v`. "
    "This means that the `config[k1][k2][k3]...[kn]` will be assigned the value `v`. "
    "Example: 'optimizer/zero_stage:0'. Then `config[optimizer][zero_stage] = 0`. "
    "If the value `v` can be interpreted as int or float, it will be casted before "
    "doing the assignment."
)


def unify(base: dict, new: dict):
    """Assigns all `key->val` mappings in `new` to `base` in-place."""
    for key, value in new.items():
        if key in base and isinstance(value, dict):
            unify(base[key], value)
        else:
            base[key] = value


def maybe_num(val: str) -> str | int | float:
    """Returns `val` as an integer or float, if it can be casted to such values, 
    other wise the original input is returned."""
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val


def override(base: dict, nested_key: list[str], val: str):
    """Overrides the nested key mapping in `base` with the specified value."""
    if len(nested_key) == 1:
        key, = nested_key
        base[key] = maybe_num(val)
    else:
        key, *rest = nested_key
        if key not in base:
            base[key] = {}
        override(base[key], rest, val)


def main(base_config_path: Path, new_config_paths: list[Path], out_path: Path,
         nnodes: Optional[int], gpus_per_node: Optional[int],
         overrides: Optional[list[str]]):
    """Unifies configs."""

    # Get base and updates it in-place with the specified new configs.
    with open(base_config_path) as f:
        base = yaml.safe_load(f)
    for new_config_path in new_config_paths:
        with open(new_config_path) as f:
            new = yaml.safe_load(f)
        unify(base, new)

    # Infers DP based on the world size, if possible.
    assert isinstance(nnodes, int) == isinstance(gpus_per_node, int), "You should provide both `--nnodes` and `--gpus-per-node`, or neither."
    if isinstance(nnodes, int):
        nproc = gpus_per_node*nnodes
        pp = base["parallelism"]["pp"]
        tp = base["parallelism"]["tp"]
        assert nproc % (pp*tp) == 0
        base["parallelism"]["dp"] = nproc//(pp*tp)

    # Final overrides.
    if isinstance(overrides, list):
        for line in overrides:
            vals = line.split(":")
            assert len(vals) == 2, "Invalid override format."
            nested_key, val = vals
            override(base, nested_key.split("/"), val)

    # Replaces `global_batch_size` to `base_config_path` and saves the final output.
    if "global_batch_size" in base["tokens"]:
        gbs = base["tokens"].pop("global_batch_size")
        mbs = base["tokens"]["micro_batch_size"]
        dp = base["parallelism"]["dp"]
        assert gbs % (dp*mbs) == 0
        base["tokens"]["batch_accumulation_per_replica"] = gbs//(dp*mbs)
    with open(out_path, "w+") as f:
        yaml.dump(base, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base-config-path", type=Path, required=True,
                        help="Path of the base nanotron configuration yaml file")
    parser.add_argument("--new-config-paths", type=Path, nargs="+", required=True,
                        help=("Paths of the new nanotron configuration yaml "
                              "files that will be used to override the default "
                              "values present in the base config path"))
    parser.add_argument("--out-path", type=Path, required=True,
                        help="Path where the unified yaml file will be written")
    parser.add_argument("--nnodes", type=int,
                        help=("Used to calculate the world size, and automatically "
                              "infer the DP size such that TP*DP*PP = WORLD_SIZE"))
    parser.add_argument("--gpus-per-node", type=int,
                        help=("Used to calculate the world size, and automatically "
                              "infer the DP size such that TP*DP*PP = WORLD_SIZE"))
    parser.add_argument("--overrides", nargs="*",
                        help=("Manually override keys in the unified config file "
                              "to a specific value. ") + OVERRIDE_FORMAT_INFO)
    args = parser.parse_args()
    main(**vars(args))
