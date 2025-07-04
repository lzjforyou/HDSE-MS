import argparse
import os
import numpy as np
import yaml

from runner import train_and_eval

def load_config(custom_fp, device_id, checkpoint_name):
    assert custom_fp, "custom_fp cannot be empty"
    assert os.path.isfile(custom_fp), custom_fp

    with open(custom_fp, "r") as custom_file:
        config_d = yaml.load(custom_file, Loader=yaml.FullLoader)

    run_name = config_d.get("run_name")
    if run_name is None:
        run_name = os.path.splitext(os.path.basename(custom_fp))[0]
        config_d["run_name"] = run_name

    data_d = config_d["data"]
    model_d = config_d["model"]
    run_d = config_d["run"]

    if device_id is not None:
        if device_id < 0:
            run_d["device"] = "cpu"
        else:
            run_d["device"] = f"cuda:{device_id}"

    if checkpoint_name:
        model_d["checkpoint_name"] = checkpoint_name

    custom_name = os.path.splitext(os.path.basename(custom_fp))[0]
    run_d["custom_name"] = custom_name

    return run_name, data_d, model_d, run_d

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device_id", type=int, required=False, help="device id (-1 for cpu)")
    parser.add_argument("-c", "--custom_fp", type=str, required=False, help="path to custom config file")
    parser.add_argument("-k", "--checkpoint_name", type=str, required=False, help="name of checkpoint to load (from checkpoint_dp)")
    parser.add_argument("-n", "--num_splits", type=int, default=1, help="number of different split seeds to run")
    flags = parser.parse_args()

    run_name, data_d, model_d, run_d = load_config(flags.custom_fp, flags.device_id, flags.checkpoint_name)

    base_split_seed = run_d["split_seed"]

    if flags.num_splits > 1:
        split_seeds = [520, 521, 522, 523, 524][:flags.num_splits]
        # split_seeds = [524][:flags.num_splits]
    else:
        split_seeds = [base_split_seed]

    print(f"Will run training {len(split_seeds)} times, using the following split_seeds: {split_seeds}")

    for i, split_seed in enumerate(split_seeds):
        print(f"\n\n===== Running training {i+1}/{len(split_seeds)}, split_seed: {split_seed} =====\n")

        run_d["split_seed"] = split_seed

        train_and_eval(data_d, model_d, run_d)

        print(f"\n===== Finished training {i+1}/{len(split_seeds)} =====\n")
    # python src/train.py -c config/nist23_P.yml
