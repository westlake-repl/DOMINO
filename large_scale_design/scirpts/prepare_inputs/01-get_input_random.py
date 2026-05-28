import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm 
import random
assert 0, "error!! check!!!"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/storage/yuanfajieLab/yuanfajie/sujin/Datasets/TED/embedding/afdb_cluster_power0.75/retrieval_results_with_seqid.tsv")
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--max_num", type=int, default=7000000)
    args = parser.parse_args()
    random.seed(0)
    query_set = set()
    with open(args.input_path, 'r') as f:
        lines = f.readlines()

    
    new_content = [lines[0]]

    lines = lines[1:]
    random.shuffle(lines)
    for line in tqdm(lines):
        line_list = line.strip().split("\t")
        query_seg, seq_id = line_list[0], float(line_list[-4])
        # if seq_id < 0.3 and query_seg not in query_set:
        if query_seg not in query_set:
            query_set.add(query_seg)
            new_content.append(line)
        if len(new_content) >= args.max_num + 1:
            break

    print(f"writing into {args.output_name}")
    with open(Path("/storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb/large_scale_design/inputs")/ args.output_name, 'w') as f:
        for line in new_content:
            f.write(line)

    
            