import os
import json
from tqdm import tqdm

## config the random sampled jsonl file and origin extract tsv file after 01-get_input.py
large_scale_input_file = "/storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb/large_scale_design/inputs/LargeScale-02-0324-retrieval_results.tsv"
sampled_jsonl_file = "/storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb/large_scale_design/results/LargeScale-02-0324-retrieval_results/LargeScale-02-0324-retrieval_results_random_5000.jsonl"


large_scale_content = open(large_scale_input_file, 'r').readlines()
header = large_scale_content[0]

res_lines = []
with open(sampled_jsonl_file, 'r') as f:
    sampled_data_list = [json.loads(line) for line in f]
    for sampled_data in sampled_data_list:
        line_idx = sampled_data['index']
        res_lines.append(large_scale_content[line_idx + 1]) 

with open("/storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb/large_scale_design/results/LargeScale-02-0324-retrieval_results/LargeScale-02-0324-retrieval_results_random_5000.tsv", 'w') as f:
    f.write(header)
    for line in res_lines:
        f.write(line)