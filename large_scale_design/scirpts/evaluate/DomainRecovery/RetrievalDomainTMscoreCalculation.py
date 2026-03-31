"""
计算预测结构与Retrieval Domain结构的TMscore

工作流程：
1. 从TSV文件读取domain序列列表
2. 通过domain序列的hash查找预先提取好的domain结构
3. 使用TMalign计算domain结构与设计结构的TMscore
4. 对每个样本，平均所有domain的TMscore
5. 保存结果到JSON

注意：
- 需要先运行 extract_retrieval_domain_structures.py 提取domain结构
- Domain结构通过MD5 hash查找
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

# TMalign 可执行文件路径
TMALIGN_EXEC = "/storage/yuanfajieLab/yuanfajie/my_project/analysis/structural_comparison/TMscore/TMalign"


def compute_domain_hash(domain_seq):
    """
    计算domain序列的MD5 hash

    参数:
    - domain_seq: domain序列（保留<unk>标记）

    返回:
    - hash值 (16进制字符串)
    """
    # 保留<unk>标记计算hash，使得有gap和无gap的domain被区分
    return hashlib.md5(domain_seq.encode()).hexdigest()



def load_all_data_from_jsonl(filepath, max_rows=-1):

    data_list = []
    with open(filepath, 'r') as f:
        for line in f:
            line_dict = json.loads(line)
            data_list.append((str(line_dict["index"]), line_dict["input_domains"], line_dict["generated_sequence"]))
    return data_list



def calculate_tmscore(pdb1, pdb2):
    """
    使用TMalign计算两个结构的TMscore

    参数:
    - pdb1: 第一个结构文件路径
    - pdb2: 第二个结构文件路径

    返回:
    - tm_score: TMscore值
    """
    random_id = random.randint(1, 1000000)
    outpath = f"/storage/yuanfajieLab/yuanfajie/tmpfile/tmp_tmalign_{random_id}.txt"

    try:
        tmalign_cmd = f"{TMALIGN_EXEC} {pdb1} {pdb2} > {outpath}"
        subprocess.call(tmalign_cmd, shell=True, timeout=60)

        # 读取输出文件获取TMscore
        with open(outpath, 'r') as f:
            content = f.readlines()

        # TMalign输出的第14行包含TMscore (索引13)
        if len(content) > 13:
            q_target_line = content[13]
            q_tm_score = float(q_target_line.split()[1])
            t_target_line = content[14]
            t_tm_score = float(t_target_line.split()[1])
            tm_score = max(q_tm_score, t_tm_score)
        else:
            print(f"警告: TMalign输出格式异常")
            tm_score = 0.0

        # 清理临时文件
        if os.path.exists(outpath):
            os.remove(outpath)

        return tm_score

    except subprocess.TimeoutExpired:
        print(f"超时: {pdb1} vs {pdb2}")
        if os.path.exists(outpath):
            os.remove(outpath)
        return 0.0
    except Exception as e:
        print(f"计算TMscore失败: {e}")
        if os.path.exists(outpath):
            os.remove(outpath)
        return 0.0


def process_single_prediction(args):
    """
    处理单个预测的Domain TMscore计算

    参数:
    - args: (idx, domain_seqs, pred_structure_path, domain_hash_to_path)

    返回:
    - (idx, mean_tm_score, domain_tm_scores): 索引、平均TMscore、所有domain的TMscore列表
    """
    idx, domain_seqs, pred_structure_path, domain_hash_to_path = args

    # 检查预测结构是否存在
    if not os.path.exists(pred_structure_path):
        print(f"警告: 预测结构不存在: {pred_structure_path}")
        return (idx, 0.0, [])

    # 对每个domain计算TMscore
    domain_tm_scores = []

    for domain_idx, domain_seq in enumerate(domain_seqs):
        # 计算domain hash
        domain_hash = compute_domain_hash(domain_seq)
        # 查找domain结构
        if domain_hash not in domain_hash_to_path:
            print(f"警告: 未找到domain结构 (hash: {domain_hash}; domain_seq {domain_seq})")
            assert 0
            domain_tm_scores.append(0.0)
            continue

        domain_pdb_path = domain_hash_to_path[domain_hash]

        if not os.path.exists(domain_pdb_path):
            print(f"警告: Domain结构文件不存在: {domain_pdb_path}")
            domain_tm_scores.append(0.0)
            continue

        # 计算TMscore
        tm_score = calculate_tmscore(domain_pdb_path, pred_structure_path)
        domain_tm_scores.append(tm_score)

    # 计算平均TMscore
    if domain_tm_scores:
        mean_tm_score = np.mean(domain_tm_scores)
    else:
        mean_tm_score = 0.0

    return (idx, mean_tm_score, domain_tm_scores)


def parse_args():
    parser = argparse.ArgumentParser(
        description='计算预测结构与Retrieval Domain结构的TMscore'
    )
    parser.add_argument(
        'jsonl_path',
        type=str,
        help='测试输出路径，包含TSV文件和预测结构'
    )
    parser.add_argument(
        '--output_structure_type',
        type=str,
        default='esmfold_results',
        choices=['esmfold_results', 'af3_output'],
        help='预测结构的类型'
    )
    parser.add_argument(
        '--domain_structures_dir',
        type=str,
        default='/storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/analysis/AFDB_structure_download/retrieval_domain_structures',
        help='Domain结构目录（包含mapping文件）'
    )
    parser.add_argument(
        '--num_processes',
        type=int,
        default=8,
        help='并行进程数'
    )
    parser.add_argument(
        '--max_rows',
        type=int,
        default=-1,
        help='最大处理行数，-1表示处理所有'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细输出'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*60)
    print("Retrieval Domain TMscore 计算")
    print("="*60)
    print(f"JSONL文件路径: {args.jsonl_path}")
    print(f"结构类型: {args.output_structure_type}")
    print(f"Domain结构目录: {args.domain_structures_dir}")
    print(f"并行进程数: {args.num_processes}")
    print("="*60)

    # 加载domain hash到结构路径的mapping
    mapping_file = Path(args.domain_structures_dir) / "domain_hash_to_structure.json"
    if not mapping_file.exists():
        print(f"错误: Mapping文件不存在: {mapping_file}")
        print("请先运行 extract_retrieval_domain_structures.py")
        return

    print("\n加载domain结构mapping...")
    with open(mapping_file, 'r') as f:
        domain_hash_to_path = json.load(f)
    print(f"已加载 {len(domain_hash_to_path)} 个domain结构映射")

    # 查找TSV文件
    if not os.path.exists(args.jsonl_path):
        print(f"错误: JSONL文件不存在: {args.jsonl_path}")
        return

    # 加载数据
    print("\n加载JSONL数据...")
    data_list = load_all_data_from_jsonl(args.jsonl_path, max_rows=args.max_rows)
    print(f"总共 {len(data_list)} 条数据")

    if not data_list:
        print("错误: 没有有效数据")
        return

    # 查找预测结构文件
    print("\n查找预测结构文件...")
    if args.output_structure_type == "esmfold_results":
        pred_structure_dir = Path(args.jsonl_path).parent / args.output_structure_type
        pred_structure_paths_dict = {str(path.name).split(".")[0]: path for path in pred_structure_dir.glob("*/*.cif")}
        # pred_structure_paths = sorted(
        #     pred_structure_dir.glob("*.pdb"),
        #     key=lambda x: int(x.stem.split("_")[-1])
        # )
    elif args.output_structure_type == "af3_output":
        pred_structure_dir = Path(args.jsonl_path).parent / args.output_structure_type
        pred_structure_paths_dict = {path.parent.name: path for path in pred_structure_dir.glob("*/*.cif")}
        # pred_structure_paths = sorted(
        #     pred_structure_dir.glob("*/*.cif"),
        #     key=lambda x: int(x.parent.name.split("_")[-1])
        # )
    else:
        print(f"错误: 不支持的结构类型: {args.output_structure_type}")
        return

    print(f"找到 {len(pred_structure_paths_dict)} 个预测结构文件")

    if len(pred_structure_paths_dict) != len(data_list):
        print(f"警告: 预测结构数量 ({len(pred_structure_paths_dict)}) 与数据行数 ({len(data_list)}) 不匹配")

    # 准备任务
    tasks = []
    for idx, domains, generated_seq in data_list:
        # if idx >= len(pred_structure_paths):
        #     print(f"警告: 索引 {idx} 超出预测结构范围")
        #     break

        tasks.append((
            idx,
            domains,
            str(pred_structure_paths_dict[idx]),
            domain_hash_to_path
        ))

    print(f"\n准备计算 {len(tasks)} 个样本的Domain TMscore...")

    # 并行计算TMscore
    with Pool(processes=args.num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_prediction, tasks),
                total=len(tasks),
                desc="Calculating Domain TMscore"
            )
        )

    # 整理结果
    mean_tm_scores = {}
    all_domain_tm_scores = {}

    for idx, mean_tm_score, domain_tm_scores in results:
        mean_tm_scores[idx] = mean_tm_score
        all_domain_tm_scores[idx] = domain_tm_scores

    # 计算统计
    overall_mean_tm_score = np.mean([s for s in list(mean_tm_scores.values()) if s > 0])

    print("\n" + "="*60)
    print("计算完成！")
    print("="*60)
    print(f"有效样本数量: {len([s for s in list(mean_tm_scores.values()) if s > 0])}")
    print(f"平均Domain TMscore: {overall_mean_tm_score:.4f}")
    print("="*60)

    # 保存结果到JSON
    output_json = args.jsonl_path.replace(".jsonl", "_log_metrics.json")

    if Path(output_json).exists():
        with open(output_json, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = {}

    metrics[f"Mean_TMscore_RetrievalDomain_{args.output_structure_type}"] = float(overall_mean_tm_score)
    # metrics[f"TMscore_RetrievalDomain_{args.output_structure_type}"] = mean_tm_scores
    metrics[f"TMscore_RetrievalDomain_AllDomains_{args.output_structure_type}"] = all_domain_tm_scores

    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"\n结果已保存到: {output_json}")


if __name__ == "__main__":
    main()
