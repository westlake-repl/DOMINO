import argparse
import json
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--tsv_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--no_header", action="store_true")
    args = parser.parse_args()

    jsonl_path = args.jsonl_path
    tsv_path = args.tsv_path

    if args.output_path is None:
        output_path = os.path.join(
            os.path.dirname(jsonl_path),
            os.path.basename(jsonl_path).replace(".jsonl", "_matched_lines.tsv")
        )
    else:
        output_path = args.output_path

    indices = []
    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)
            indices.append(data["index"])

    unique_indices = set(indices)
    idx2line = {}

    with open(tsv_path, "r") as f:
        header = f.readline()
        for line_idx, line in enumerate(f):
            if line_idx in unique_indices:
                idx2line[line_idx] = line

    missing_indices = [idx for idx in indices if idx not in idx2line]
    if len(missing_indices) > 0:
        raise ValueError(f"These indices are missing in tsv: {missing_indices[:10]}")

    with open(output_path, "w") as f:
        if not args.no_header:
            f.write(header)
        for idx in indices:
            f.write(idx2line[idx])

    print(f"Extracted {len(indices)} lines to {output_path}")
