from bloom_filter2 import BloomFilter
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import os

INPUT_DIR = os.path.join(".", "input")      # Directory containing data to analyse
JSON_DIR = os.path.join(".", "json")        # Directory where BitReps writes intermediary JSON


def dir_setup():
    """
    Create the necessary directories if they don't already exist
    :return: None
    """
    if not os.path.exists(INPUT_DIR):
        os.mkdir(INPUT_DIR)
    if not os.path.exists(JSON_DIR):
        os.mkdir(JSON_DIR)


def analyse_data(blocks, bs, fpr):
    """
    Create a Bloom filter and populate it with each encountered block, whilst maintaining frequency information in a
    dictionary
    :param blocks: List of integers representing blocks in analysed file
    :param bs: Blocksize, in bytes
    :param fpr: User-specified maximum false positive rate of the Bloom filter
    :return: The Bloom filter object and the frequency dictionary
    """
    freqs = {
        "num_blocks": len(blocks),
        "blocksize": bs * 8,
        "fpr": fpr,
        "blocks": defaultdict(lambda: 1)
    }

    bf = BloomFilter(max_elements=len(blocks), error_rate=fpr)
    for b in tqdm(blocks):
        if b not in bf:
            # This is the first time we've seen this block
            bf.add(b)
        else:
            # Increment the block's frequency
            freqs["blocks"][b] += 1

    return bf, freqs


def get_blocks(fp, bs):
    """
    Open file at fp, split into blocks of size bs, return as list of integers representing blocks
    :param fp: Path of file to analyse
    :param bs: Blocksize in bytes
    :return: List of integers representing each block in analysed file
    """
    blocks = []
    with open(fp, "rb") as f:
        while True:
            block = f.read(bs)
            if not block:
                # We've run out of data
                break
            blocks.append(int.from_bytes(block, "big"))
    return blocks


def gather_inputs():
    """
    Gather each file in INPUT_DIR and return as a list of file paths
    :return: List of file paths
    """
    inputs = []
    for root, dirs, files in os.walk(INPUT_DIR):
        for f in files:
            inputs.append(os.path.join(root, f))
    return inputs


def main():
    # Directory setup
    dir_setup()

    # Deal with user input
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-fp", "--file_path", help="Path to input data")
    group.add_argument("-b", "--batch", help="Analyse all input files", action="store_const", const=gather_inputs())
    parser.add_argument("bs", help="Blocksize", type=int)
    parser.add_argument("fpr", help="Desired false positive rate")
    args = parser.parse_args()

    # Validate user input
    bs = int(args.bs / 8)
    fpr = float(args.fpr)
    assert 0 < fpr < 1

    # Gather and validate input file paths
    inputs = []
    if args.file_path:
        assert os.path.exists(args.file_path)
        inputs.append(args.file_path)
    else:
        for fp in args.batch:
            assert os.path.exists(fp)
            inputs.append(fp)

    for input_file in tqdm(inputs):
        # Split input data into hex blocks
        blocks = get_blocks(input_file, bs)

        # Create a Bloom filter and associated occurrence dictionary
        bloom, freqs = analyse_data(blocks, bs, fpr)

        # Write dictionary as JSON file
        fn = Path(input_file).stem
        json_path = os.path.join(JSON_DIR, f"{fn}_{bs * 8}_{fpr}.json")
        with open(json_path, "w") as f:
            json.dump(freqs, f, indent=2)


if __name__ == "__main__":
    main()
