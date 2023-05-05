from scipy.stats import chisquare
from collections import Counter
import numpy as np
import argparse
import logging
import json
import sys
import os

JSON_DIR = os.path.join(".", "json")
OUTPUT_DIR = os.path.join(".", "output")
MIN_FREQ = 5


def merge_buckets(obs, exp):
    obs_merged = []
    exp_merged = []

    i = 0
    while i < len(exp) and exp[i] >= MIN_FREQ:
        exp_merged.append(exp[i])
        obs_merged.append(obs[i])
        i += 1
    if len(exp) != len(exp_merged):
        if i == len(exp) - 1:
            # This was the final expected bucket, so there are no subsequent buckets to merge with
            # Therefore, merge with the penultimate bucket
            exp_merged[-1] += exp[i]
            obs_merged[-1] += obs[i]
        else:
            # There are still buckets after this bucket, so merge with them all
            exp_merged.append(sum(exp[i:]))
            obs_merged.append(sum(obs[i:]))
            if exp_merged[-1] < MIN_FREQ:
                exp_merged[-2] += exp_merged[-1]
                exp_merged = exp_merged[:-1]
                obs_merged[-2] += obs_merged[-1]
                obs_merged = obs_merged[:-1]

    assert len(exp_merged) == len(obs_merged)

    return obs_merged, exp_merged


def prep_frequency_distributions(obs, exp):
    all_keys = set(obs.keys()).union(set(exp.keys()))
    obs_freq = []
    exp_freq = []

    for k in all_keys:
        obs_freq.append(obs.get(k, 0) * k)
        exp_freq.append(exp.get(k, 0) * k)

    return obs_freq, exp_freq


def main():
    # Set up debugger
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    # Deal with user input
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_fp", help="File path of expected data")
    parser.add_argument("obs_fp", help="File path of observed data")
    args = parser.parse_args()

    # Validate user input
    assert os.path.exists(args.obs_fp)
    assert os.path.exists(args.exp_fp)

    # Read inputs into arrays representing frequency distributions
    with open(args.obs_fp, "r") as f:
        obs_freqs_data = json.load(f)
    with open(args.exp_fp, "r") as f:
        exp_freqs_data = json.load(f)

    obs_freqs = np.array(list(obs_freqs_data["blocks"].values()))
    exp_freqs = np.array(list(exp_freqs_data["blocks"].values()))

    # Pad the frequency distributions to account for unique blocks
    num_blocks = exp_freqs_data["num_blocks"]
    pad_amount_obs = num_blocks - np.sum(obs_freqs)
    pad_amount_exp = num_blocks - np.sum(exp_freqs)

    padded_obs = np.pad(obs_freqs, (0, pad_amount_obs), "constant", constant_values=(0, 1))
    padded_exp = np.pad(exp_freqs, (0, pad_amount_exp), "constant", constant_values=(0, 1))

    # Make sure everything looks good so far
    assert np.sum(padded_exp) == np.sum(padded_obs) == num_blocks

    # Convert data into format acceptable for goodness-of-fit test
    obs_counter = sorted(Counter(padded_obs).items())
    exp_counter = sorted(Counter(padded_exp).items())

    obs, exp = prep_frequency_distributions(dict(obs_counter), dict(exp_counter))

    obs_merged, exp_merged = merge_buckets(obs, exp)

    logging.info(f"Observed Before Merge: {obs}")
    logging.info(f"Expected Before Merge: {exp}")

    logging.info(f"Observed Merged: {obs_merged}")
    logging.info(f"Expected Merged: {exp_merged}")

    logging.info(f"Result: {chisquare(obs_merged, exp_merged)}")


if __name__ == "__main__":
    main()
