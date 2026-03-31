import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
from multiprocessing import Pool
from Bio import Align
import lmdb

def get_seq(name, txn):
    return txn.get(name.encode()).decode()

def calculateseqid(seq1, seq2):
    """
    使用Bio.Align计算两个序列的sequence identity
    使用全局比对，标准的seq_id计算方式

    参数:
    - seq1: 第一个序列（通常是GT序列）
    - seq2: 第二个序列（通常是预测序列）

    返回:
    - identity_pct: 序列一致性百分比 (0-1之间的浮点数)
    """
    if len(seq1) == 0 or len(seq2) == 0:
        return 0.0

    # 创建比对器
    aligner = Align.PairwiseAligner()

    # 使用全局比对（默认模式）
    aligner.mode = 'global'

    # 进行比对
    alignments = aligner.align(seq1, seq2)

    try:
        best_alignment = alignments[0]
    except IndexError:
        return 0.0

    # 提取比对信息
    counts = best_alignment.counts()
    matches = counts.identities
    # 比对长度 = 匹配 + 错配 + 所有空位
    alignment_length = counts.aligned + counts.gaps

    identity_pct = (matches / alignment_length) if alignment_length > 0 else 0.0

    return identity_pct




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--generation_path", type=str, default="/storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb/large_scale_design/inputs/LargeScale-01-0301-retrieval_results.tsv")
    parser.add_argument("--num_processes", type=int, default=32, help="Number of processes for parallel computation")
    args = parser.parse_args()

    target_path = Path(args.jsonl_path).parent
    generation_path = args.generation_path

    all_generation_content = open(generation_path, 'r').readlines()
    idx2seq = {}
    with open(args.jsonl_path, 'r') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            idx2seq[data["index"]] = "".join(data["input_domains"])
    print("mean of len seqs", np.mean([len(x) for x in idx2seq.values()]))

    # print(idx2structure_path)
    # print(idx2structure_path[4121248])
    # print(all_generation_content[4121248])
    # print(all_generation_content[4121247])

    # print(all_generation_content[4121249])
    env = lmdb.open("/storage/yuanfajieLab/yuanfajie/datasets/AFDB/LMDB_seqonly/", readonly=True, lock=False)
    txn = env.begin()

    # assert 0
    def process_single_pair(idx):
        target_line = all_generation_content[idx + 1].strip().split("\t")
        query_uid = target_line[-3].split(",")
        target_uid = target_line[-1].split(",")
        query_seq_list = [get_seq(q_uid, txn) for q_uid in query_uid]
        target_seq_list = [get_seq(t_uid, txn) for t_uid in target_uid]
        seqid_query_list = []
        seqid_target_list = []
        for q_seq in query_seq_list:
            seqid_query_list.append(calculateseqid(idx2seq[idx], q_seq))
        for t_seq in target_seq_list:
            seqid_target_list.append(calculateseqid(idx2seq[idx], t_seq))
        
        seqid_query = np.max(seqid_query_list)
        seqid_target = np.max(seqid_target_list)
        return seqid_query, seqid_target

    # tasks = [(idx, query_uid, target_uid) for idx, (query_uid, target_uid) in enumerate(uid_list)]
    tasks = list(idx2seq.keys())

    with Pool(processes=args.num_processes) as pool:
        results = list(tqdm(pool.imap(process_single_pair, tasks), total=len(tasks), desc="calculating seq id"))

    seqid_query_structure_dict = {idx: r[0] for idx, r in zip(tasks, results)}
    seqid_target_structure_dict = {idx: r[1] for idx, r in zip(tasks, results)}
    print(f"Mean_seqid_query", np.mean(list(seqid_query_structure_dict.values())))
    print(f"Mean_seqid_target", np.mean(list(seqid_target_structure_dict.values())))

    # metrics = json.load(open(Path(target_path) / "log_metrics.json", 'r'))
    # metrics[f"Mean_seqid_query"] = np.mean(list(seqid_query_structure_dict.values()))
    # metrics[f"Mean_seqid_target"] = np.mean(list(seqid_target_structure_dict.values()))
    # metrics[f"seqid_query"] = seqid_query_structure_dict
    # metrics[f"seqid_target"] = seqid_target_structure_dict
    # with open(Path(target_path) / "log_metrics.json", 'w') as f:
    #     json.dump(metrics, f, indent=4)