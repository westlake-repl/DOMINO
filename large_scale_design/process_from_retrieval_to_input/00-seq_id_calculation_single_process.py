import lmdb
from tqdm import tqdm
from Bio.Align import PairwiseAligner
from multiprocessing import Pool, Manager
import os

assert 0, "error!! check!!!"
# 配置参数
INPUT_TSV_PATH = "/storage/yuanfajieLab/yuanfajie/sujin/Datasets/TED/embedding/afdb_cluster_power0.75/retrieval_results.tsv"
# INPUT_TSV_PATH = "tmp.tsv"
OUTPUT_TSV_PATH = "/storage/yuanfajieLab/yuanfajie/sujin/Datasets/TED/embedding/afdb_cluster_power0.75/retrieval_results_with_seqid.tsv"
LMDB_SEQONLY_PATH = "/storage/yuanfajieLab/yuanfajie/datasets/AFDB/LMDB_seqonly"
LMDB_DOMAIN_PATH = "/storage/yuanfajieLab/yuanfajie/datasets/AFDB/LMDB_uid2domain_info"
NUM_PROCESSES = 1  # 使用所有可用CPU核心


def clear_seq(seq: str) -> str:
    return seq[::2]


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


# 全局变量用于存储每个进程的LMDB连接
txn = None
domain_txn = None


def init_worker():
    """初始化每个进程的LMDB连接(只调用一次)"""
    global txn, domain_txn

    env = lmdb.open(LMDB_SEQONLY_PATH, readonly=True, lock=False)
    txn = env.begin()

    domain_env = lmdb.open(LMDB_DOMAIN_PATH, readonly=True, lock=False)
    domain_txn = domain_env.begin()


def process_line(line_data):
    """处理单行数据,计算seq id"""
    global txn, domain_txn

    line, line_idx = line_data

    # try:
    fields = line.strip().split("\t")
    query_seq, target_seq = fields[:2]
    gt_uids = fields[-3]
    gt_uid = gt_uids.split(",")[0]

    # 从LMDB获取数据
    try:
        gt_seq = txn.get(gt_uid.encode()).decode()
        gt_domain_info_list = eval(domain_txn.get(gt_uid.encode()).decode())
    except:
        print(f"从LMDB获取数据时出错: {gt_uid}")
        assert 0
    query_seq_clean = query_seq.replace("<unk>", "")
    target_seq_clean = target_seq.replace("<unk>", "")

    # 计算所有domain的seq id
    seq_id_list = []
    for domain_info in gt_domain_info_list:
        cur_domain_seq = domain_fillin(gt_seq, domain_info)
        if cur_domain_seq == clear_seq(query_seq_clean):
            continue
        seq_id_list.append(calc_seq_identity(clear_seq(target_seq_clean), cur_domain_seq))

    # 计算最大seq id
    print("seq_id_list", seq_id_list)
    max_seq_id = max(seq_id_list) if seq_id_list else 0.0
    print("max_seq_id", max_seq_id)
    # 更新第3列(索引为2)的seq id值
    fields[2] = str(max_seq_id)
    new_line = "\t".join(fields) + "\n"

    return line_idx, new_line

    # except Exception as e:
    #     print(f"处理第{line_idx}行时出错: {e}")
    #     return line_idx, line  # 出错时返回原始行


def main():
    print(f"使用 {NUM_PROCESSES} 个进程处理文件...")
    print(f"输入文件: {INPUT_TSV_PATH}")
    print(f"输出文件: {OUTPUT_TSV_PATH}")

    init_worker()
    # 读取所有行
    with open(INPUT_TSV_PATH, 'r') as f:
        header = f.readline()
        # lines = f.readlines()
        for idx, line in enumerate(f):
            line_idx, new_line = process_line((line, idx))
            print(new_line)
            break
1

    # # 写入新文件
    # print(f"写入结果到: {OUTPUT_TSV_PATH}")
    # with open(OUTPUT_TSV_PATH, 'w') as f:
    #     f.write(header)
    #     for _, processed_line in results:
    #         f.write(processed_line)

    # print("处理完成!")


if __name__ == "__main__":
    main()