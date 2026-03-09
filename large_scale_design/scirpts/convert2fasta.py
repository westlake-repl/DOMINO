import argparse
import json
import os
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb/large_scale_design/results/0209-retrieval_results-TED-650M-plddt70-sameQuery.jsonl")
    args = parser.parse_args()

    input_path = args.input_path
    output_path = os.path.join(os.path.dirname(input_path), os.path.basename(input_path).replace(".jsonl", ".fasta"))
    with open(input_path, "r") as f:
        for line in f:
            data = json.loads(line)
            index = data["index"]
            generated_sequence = data["generated_sequence"]
            with open(output_path, "a") as f:
                f.write(f">{index}\n")
                f.write(f"{generated_sequence}\n")