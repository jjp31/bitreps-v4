from bloom_filter2 import BloomFilter
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import os

INPUT_DIR = os.path.join(".", "input")
JSON_DIR = os.path.join(".", "json")
OUTPUT_DIR = os.path.join(".", "output")


def dir_setup():
    if not os.path.exists(INPUT_DIR):
        os.mkdir(INPUT_DIR)
    if not os.path.exists(JSON_DIR):
        os.mkdir(JSON_DIR)
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)


def analyse_data(blocks, bs, fpr):
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
    blocks = []
    with open(fp, "rb") as f:
        while True:
            block = f.read(bs)
            if not block:
                # We've run out of data
                break
            blocks.append(int.from_bytes(block, "big"))
    return blocks


def main():
    # Directory setup
    dir_setup()

    # Deal with user input
    parser = argparse.ArgumentParser()
    parser.add_argument("fp", help="Path to input data")
    parser.add_argument("bs", help="Blocksize", type=int)
    parser.add_argument("fpr", help="Desired false positive rate")
    args = parser.parse_args()

    # Validate user input
    bs = int(args.bs / 8)
    fpr = float(args.fpr)
    assert os.path.exists(args.fp)
    assert 0 < fpr < 1

    # Split input data into hex blocks
    blocks = get_blocks(args.fp, bs)

    # Create a Bloom filter and associated occurrence dictionary
    bloom, freqs = analyse_data(blocks, bs, fpr)

    # Write dictionary as JSON file
    fn = Path(args.fp).stem
    json_path = os.path.join(JSON_DIR, f"{fn}_{bs * 8}_{fpr}.json")
    with open(json_path, "w") as f:
        json.dump(freqs, f, indent=2)


if __name__ == "__main__":
    main()
