import argparse
import glob
import json
import os
from tqdm import tqdm

import numpy as np
from Bio import Align


def calculate_piece_identity(piece, target_seq):
    """
    计算单个domain片段与目标序列的最佳匹配identity

    参数:
    - piece: domain片段（不含<unk>）
    - target_seq: 目标序列

    返回:
    - identities: 匹配的氨基酸数量
    """
    if len(piece) == 0:
        return 0

    # 创建比对器
    aligner = Align.PairwiseAligner()

    # 使用局部比对，自动找到最佳匹配区域
    aligner.mode = 'local'

    # 使用默认的gap惩罚（这样中间有插入/缺失会被惩罚）
    # 不设置为0，避免ABC匹配AXXBXXC得到1.0的问题

    # 进行比对
    alignments = aligner.align(piece, target_seq)

    try:
        best_alignment = alignments[0]
    except IndexError:
        # 如果没有比对结果
        return 0

    # 提取匹配的氨基酸数量
    counts = best_alignment.counts()
    return counts.identities


def remove_unk_tokens(domain):
    """
    从domain中移除<unk>标记

    参数:
    - domain: 包含<unk>的domain字符串

    返回:
    - 移除<unk>后的序列
    """
    return domain.replace("<unk>", "")


def split_domain_by_unk(domain):
    """
    按<unk>分割domain

    参数:
    - domain: 包含<unk>的domain字符串

    返回:
    - pieces: domain片段列表（不含<unk>）
    """
    pieces = [piece for piece in domain.split("<unk>") if piece]
    return pieces


def domain_seqid(domain, target_seq):
    """
    计算domain与目标序列的sequence identity

    策略：
    1. 按<unk>分割domain成多个片段
    2. 对每个片段独立进行局部比对（使用正常的gap惩罚）
    3. 汇总所有片段的匹配数
    4. seq_id = 总匹配数 / domain总长度（去除<unk>后）

    这样：
    - 如果domain有<unk>: ABC<unk>DEF，各片段可以在序列不同位置匹配 ✅
    - 如果domain没有<unk>: ABC，必须连续匹配，中间插入会被gap惩罚 ✅

    参数:
    - domain: domain序列（可能包含<unk>）
    - target_seq: 目标序列

    返回:
    - seqid: sequence identity (0-1之间的浮点数)
    """
    # 按<unk>分割
    pieces = split_domain_by_unk(domain)

    if not pieces:
        return 0.0

    # 计算domain总长度（不含<unk>）
    total_domain_length = sum(len(piece) for piece in pieces)

    if total_domain_length == 0:
        return 0.0

    # 对每个片段计算匹配数
    total_identities = 0
    for piece in pieces:
        identities = calculate_piece_identity(piece, target_seq)
        total_identities += identities

    # 计算seq_id
    seqid = total_identities / total_domain_length

    return seqid


def calculate_domain_matching_metrics(domains, pred_seq, thresholds=[0.9, 0.7, 0.5, 0.3]):
    """
    计算domain匹配的各项指标

    参数:
    - domains: domain列表
    - pred_seq: 预测序列
    - thresholds: 要统计的阈值列表

    返回:
    - metrics: 包含各项指标的字典
    """
    domain_seqids = []
    domain_match_info = []

    # 计算每个domain的seq_id
    for i, domain in enumerate(domains):
        seqid = domain_seqid(domain, pred_seq)
        domain_seqids.append(seqid)
        domain_match_info.append({
            'domain_idx': i,
            'domain': domain,
            'clean_domain': remove_unk_tokens(domain),
            'seqid': seqid
        })

    # 计算平均seq_id
    mean_seqid = np.mean(domain_seqids) if domain_seqids else 0.0

    # 计算各阈值下的匹配率
    threshold_metrics = {}
    for threshold in thresholds:
        matched_count = sum(1 for seqid in domain_seqids if seqid >= threshold)
        match_ratio = matched_count / len(domains) if len(domains) > 0 else 0.0
        threshold_metrics[f'match_ratio_at_{int(threshold*100)}'] = match_ratio

    metrics = {
        'mean_domain_seqid': mean_seqid,
        'total_domains': len(domains),
        **threshold_metrics,
        'domain_details': domain_match_info,
        "domain_seqids": domain_seqids
    }

    return metrics


def load_all_data_from_jsonl(filepath, max_rows=-1):

    data_list = []
    with open(filepath, 'r') as f:
        for line in f:
            line_dict = json.loads(line)
            data_list.append((line_dict["index"], line_dict["input_domains"], line_dict["generated_sequence"]))
    return data_list


def parse_args():
    parser = argparse.ArgumentParser(
        description='计算domain匹配的软指标（基于sequence identity）'
    )
    parser.add_argument("res_jsonl_path", type=str)
    parser.add_argument("--thresholds", type=float, nargs='+',
                       default=[0.9, 0.7, 0.5, 0.3],
                       help="要统计的seq_id阈值列表")
    parser.add_argument("--max_rows", type=int, default=-1,
                       help="最大处理行数，-1表示处理所有行")
    parser.add_argument("--verbose", action='store_true',
                       help="显示详细输出")
    return parser.parse_args()


def main():
    args = parse_args()

    res_jsonl_path = args.res_jsonl_path

    if not os.path.exists(res_jsonl_path):
        print(f"警告:  {res_jsonl_path} 不存在")
        return

    # 用于汇总所有文件的指标
    all_mean_seqids = []
    all_threshold_matches = {f'match_ratio_at_{int(t*100)}': []
                            for t in args.thresholds}

    if args.verbose:
        print(f"\n处理文件: {os.path.basename(res_jsonl_path)}")

    data_list = load_all_data_from_jsonl(res_jsonl_path, max_rows=args.max_rows)

    if not data_list:
        print(f"警告: {res_jsonl_path} 中没有有效数据")
        exit(0)

    # 对每一行计算指标
    file_mean_seqids = []
    file_all_seqids = {}
    file_threshold_matches = {f'match_ratio_at_{int(t*100)}': []
                                for t in args.thresholds}

    for entry_id, domains, pred_seq in tqdm(data_list):
        metrics = calculate_domain_matching_metrics(domains, pred_seq, args.thresholds)
        file_mean_seqids.append(metrics['mean_domain_seqid'])
        file_all_seqids[entry_id] = metrics["domain_seqids"]

        for key in file_threshold_matches.keys():
            file_threshold_matches[key].append(metrics[key])

    # 汇总当前文件的结果
    all_mean_seqids.extend(file_mean_seqids)
    for key in all_threshold_matches.keys():
        all_threshold_matches[key].extend(file_threshold_matches[key])

    if args.verbose:
        print(f"  文件平均 seq_id: {np.mean(file_mean_seqids):.4f}")
        for key, values in file_threshold_matches.items():
            print(f"  文件平均 {key}: {np.mean(values):.4f}")

    # 计算整体平均指标
    overall_mean_seqid = np.mean(all_mean_seqids) if all_mean_seqids else 0.0
    overall_threshold_metrics = {}
    for key, values in all_threshold_matches.items():
        overall_threshold_metrics[key] = np.mean(values) if values else 0.0

    # 打印结果
    print("\n" + "="*60)
    print(f"测试输出路径: {res_jsonl_path}")
    print(f"处理的样本总数: {len(all_mean_seqids)}")
    print(f"\n整体平均 domain seq_id: {overall_mean_seqid:.4f}")
    print("\n不同阈值下的匹配率:")
    for threshold in sorted(args.thresholds, reverse=True):
        key = f'match_ratio_at_{int(threshold*100)}'
        print(f"  seq_id >= {threshold:.0%}: {overall_threshold_metrics[key]:.4f} ({overall_threshold_metrics[key]:.2%})")
    print("="*60)

    # 保存结果到JSON
    output_json = res_jsonl_path.replace(".jsonl", "_log_metrics.json")
    res_dict = {
        'mean_domain_seqid': overall_mean_seqid,
        "seqid_Domain_AllDomains": file_all_seqids,
        **overall_threshold_metrics
    }

    # 如果已存在log_metrics.json，合并结果
    if os.path.exists(output_json):
        with open(output_json, "r") as f:
            prev_metrics = json.load(f)
        prev_metrics.update(res_dict)
        res_dict = prev_metrics

    with open(output_json, "w") as f:
        json.dump(res_dict, f, indent=2)

    print(f"\n结果已保存到: {output_json}")




if __name__ == "__main__":
    main()
