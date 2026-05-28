import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm 
import random
from collections import defaultdict
from Bio.Align import PairwiseAligner
import lmdb

sa2aa = lambda sa: "".join([i[::2] for i in  sa.split("<unk>")])

LMDB_SEQONLY_PATH = "/storage/yuanfajieLab/yuanfajie/datasets/AFDB/LMDB_seqonly"
LMDB_DOMAIN_PATH = "/storage/yuanfajieLab/yuanfajie/datasets/AFDB/LMDB_uid2domain_info"

name2seq_env = lmdb.open(LMDB_SEQONLY_PATH, readonly=True, lock=False)
name2domain_env = lmdb.open(LMDB_DOMAIN_PATH, readonly=True, lock=False)

name2seq_txn = name2seq_env.begin()
name2domain_txn = name2domain_env.begin()


def domain_fillin(sequence: str, domain_info: str) -> str:
    """
    Fill in the domain information into the sequence. If the domain is not continuous, link the domain with <unk> tokens.
    input:
        sequence: str, e.g. "MALWMRLLPLLALLALWGPDPAAAPSL"
        domain_info: str, e.g. "1-3_5-10", start from 1
    output:
        domain_filled_sequence: str, e.g. "MAL<unk>MRLLPL"
    """
    domain_info = domain_info.split("_")
    domain_info = [tuple(map(int, domain.split("-"))) for domain in domain_info]
    domain_info = sorted(domain_info, key=lambda x: x[0])
    domain_filled_sequence = []
    for domain in domain_info:
        start, end = domain
        domain_filled_sequence.append(sequence[start - 1 : end])
    return "".join(domain_filled_sequence)


def name2domain(name):
    domain_info = name2domain_txn.get(name.encode("utf-8"))
    domain_info_list = json.loads(domain_info.decode("utf-8"))
    seq = name2seq(name)

    return [domain_fillin(seq, domain_info) for domain_info in domain_info_list]

def name2seq(name):
    seq = name2seq_txn.get(name.encode("utf-8"))
    if seq is not None:
        return seq.decode("utf-8")
    else:
        return None


def calc_seq_identity(seq1: str, seq2: str) -> float:
    """计算蛋白质序列相似度"""
    try:
        aligner = PairwiseAligner()
        aligner.mode = "local"

        alignment = next(aligner.align(seq1, seq2))
        a1, a2 = alignment
        identity = sum(1 for a, b in zip(a1, a2) if a == b) / len(a1)
        return identity

    except Exception as e:
        print(f"计算序列相似度时出错: {e}")
        return 0.0

def get_gt_target_aa_list(uniprot_id_list, query_seg_aa):
    gt_target_aa_list = []
    for uniprod_id in uniprot_id_list:
        domain_aa_list = name2domain(uniprod_id)
        for domain_aa in domain_aa_list:
            if domain_aa != query_seg_aa:
                gt_target_aa_list.append(domain_aa)
    return gt_target_aa_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb/large_scale_design/inputs/LargeScale-02-0324-retrieval_results_random5000_backup.tsv")
    parser.add_argument("--output_name", type=str, default="/storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb/large_scale_design/inputs/LargeScale-02-0324-retrieval_results_random_5000_seqid_lt_0.3.tsv")
    args = parser.parse_args()
    random.seed(0)
    with open(args.input_path, 'r') as f:
        lines = f.readlines()

    
    new_content = [lines[0]]
    lines = lines[1:]

    qeury_seg_to_lines = defaultdict(list)
    for line in tqdm(lines):
        line_list = line.strip().split("\t")
        query_seg = line_list[0]
        qeury_seg_to_lines[query_seg].append(line)

        
    for query_seg in tqdm(qeury_seg_to_lines.keys()):
        lines = qeury_seg_to_lines[query_seg]
        for line in lines:
            line_list = line.strip().split("\t")
            target_seg = line_list[1]
            query_seq_aa = sa2aa(query_seg)
            target_seq_aa = sa2aa(target_seg)

            gt_uniprot_ids = line_list[-3].split(",")
            gt_target_aa_list = get_gt_target_aa_list(gt_uniprot_ids, query_seq_aa)
            seq_id_list = [
                calc_seq_identity(target_seq_aa, gt_target_aa)
                for gt_target_aa in gt_target_aa_list
            ]
            seq_id = max(seq_id_list)
            if seq_id < 0.3:
                new_content.append(line)
                break
            # seq_id = calc_seq_identity(query_seq_aa, target_seq_aa)
            # if seq_id < 0.3:
            #     new_content.append(line)
            #     break

    print(f"writing into {args.output_name}")
    with open(args.output_name, 'w') as f:
        for line in new_content:
            f.write(line)

    
            