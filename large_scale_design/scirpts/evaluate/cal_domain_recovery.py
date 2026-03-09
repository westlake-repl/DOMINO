import argparse
import glob
import json
import os

import numpy as np
import sys
sys.path.append("/storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/analysis/domain_matching")
from hard_domain_matching import find_best_domain_match
from tqdm import tqdm


def load_all_data_from_jsonl(filepath, max_rows: int = -1):
    """
    从TSV文件加载所有数据

    参数:
    - filepath: TSV文件路径，格式为 domain_list, gt_seqs, pred_seqs

    返回:
    - data_list: 包含所有行的数据列表，每行为 (domains, gt_seq, pred_seq)
    """
    data_list = []

    with open(filepath, "r") as f:
        lines = f.readlines()  # skip the header
    if max_rows != -1:
        lines = lines[:max_rows]
    for line_idx, line in enumerate(lines):


        data = json.loads(line) 
        domains = data["input_domains"]
        pred_seq = data["generated_sequence"]

        data_list.append((domains, pred_seq))

    return data_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("res_jsonl_path", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    mean_finded_ratio = []


    data_list = load_all_data_from_jsonl(args.res_jsonl_path)
    for domains, pred_seq in tqdm(data_list):
        domain_piece_results, domain_results = find_best_domain_match(
            domains, pred_seq
        )
        mean_finded_ratio.append(np.mean(domain_results))
    
    
    print(
        "{} mean_matched_domain_ratio, {:.2f}".format(
            args.res_jsonl_path, np.mean(mean_finded_ratio)
        )
    )


if __name__ == "__main__":
    main()
