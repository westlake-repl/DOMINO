import argparse
import math
from pathlib import Path
from tqdm import tqdm 
import random
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/storage/yuanfajieLab/yuanfajie/sujin/Datasets/TED/embedding/afdb_cluster_power0.75/retrieval_results_with_tmscore.tsv")
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--prefetched_file", type=str, default=None)
    parser.add_argument("--max_num", type=int, default=5000000)
    parser.add_argument("--tmscore_column", type=str, default="Max_TMscore")
    parser.add_argument("--tmscore_threshold", type=float, default=0.5)
    args = parser.parse_args()
    random.seed(0)
    with open(args.input_path, 'r') as f:
        lines = f.readlines()

    
    new_content = [lines[0]]
    header = lines[0].strip().split("\t")
    if args.tmscore_column not in header:
        raise ValueError(f"{args.tmscore_column} is not found in {args.input_path}")
    tmscore_idx = header.index(args.tmscore_column)
    lines = lines[1:]

    ## first get the query_seg to lines mapping
    qeury_seg_to_lines = defaultdict(list)
    for line in tqdm(lines):
        line_list = line.strip().split("\t")
        query_seg = line_list[0]
        qeury_seg_to_lines[query_seg].append(line)

    # random select the query segments
    if args.prefetched_file is None:
        query_segs = list(qeury_seg_to_lines.keys())
        random.shuffle(query_segs)
        query_segs = query_segs[:args.max_num]
    else:
        with open(args.prefetched_file, 'r') as f:
            query_segs = [line.strip().split("\t")[0] for line in f.readlines()[1:]]

    for query_seg in query_segs:
        lines = qeury_seg_to_lines[query_seg]
        for line in lines:
            line_list = line.strip().split("\t")
            tmscore = float(line_list[tmscore_idx])
            if math.isfinite(tmscore) and tmscore < args.tmscore_threshold:
                new_content.append(line)
                break

    print(f"writing into {args.output_name}")
    with open(Path("/storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb/large_scale_design/inputs")/ args.output_name, 'w') as f:
        for line in new_content:
            f.write(line)
