import argparse
import random
import numpy as np
from pathlib import Path
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    input_path = args.input_path
    assert input_path.endswith(".jsonl")
    with open(input_path, "r") as f:
        lines = f.readlines()
    random.shuffle(lines)
    lines = lines[:args.num_samples]

    input_basename = Path(input_path).stem
    output_dir= Path(input_path).parent / input_basename 
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_basename}_random_{args.num_samples}.jsonl"
    print(f"Writing to {output_path}")
    with open(output_path, "w") as f:
        for line in lines:
            f.write(line)
