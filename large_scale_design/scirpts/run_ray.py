import argparse
from omegaconf import OmegaConf
import logging
import torch
from transformers import EsmTokenizer
from tqdm import tqdm
import ray
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import defaultdict
import json
import random

import sys
import os

# 获取项目根目录的绝对路径
PROJECT_ROOT = "/storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb"
sys.path.append(os.path.join(PROJECT_ROOT, "src/DomainComb"))
sys.path.append(PROJECT_ROOT)
from large_scale_design.scirpts.dataloader import Step2Dataset
from utils.init_utils import construct_class_by_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """
    设置所有随机数生成器的种子以确保可重复性
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保 CUDA 操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"随机种子已设置为: {seed}")


def sort_and_batch_by_length(dataset: Step2Dataset, batch_size: int) -> List[List[Tuple[int, Dict]]]:
    """
    按输入长度排序后直接分batch

    Args:
        dataset: 数据集
        batch_size: batch大小

    Returns:
        List of batches, each batch is List[(idx, sample)]
    """
    logger.info("按长度排序数据...")

    # 计算每个样本的总长度
    lengths = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        total_len = sum(len(domain) for domain in sample["domain"])
        lengths.append((idx, total_len))

    # 按长度排序
    lengths.sort(key=lambda x: x[1])

    logger.info(f"长度范围: [{lengths[0][1]}, {lengths[-1][1]}]")

    # 按顺序分batch
    batches = []
    for i in range(0, len(lengths), batch_size):
        batch_indices = [idx for idx, _ in lengths[i:i + batch_size]]
        batch = [(idx, dataset[idx]) for idx in batch_indices]
        batches.append(batch)

    logger.info(f"共创建 {len(batches)} 个batches")
    return batches


@ray.remote(num_gpus=1)
class InferenceWorker:
    """
    Ray worker，每个worker占用一张GPU
    """
    def __init__(self, config_path: str, checkpoint_path: str, gpu_id: int, seed: int = None):
        # 在worker进程中设置路径（Ray worker是独立进程）
        import sys
        import os
        import random
        import numpy as np
        import torch
        
        PROJECT_ROOT = "/storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb"
        if os.path.join(PROJECT_ROOT, "src/DomainComb") not in sys.path:
            sys.path.append(os.path.join(PROJECT_ROOT, "src/DomainComb"))
        if PROJECT_ROOT not in sys.path:
            sys.path.append(PROJECT_ROOT)

        # 导入必要的模块
        from utils.init_utils import construct_class_by_name

        # 设置随机种子（如果提供）- 直接在worker内部设置，避免序列化问题
        if seed is not None:
            worker_seed = seed + gpu_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            torch.cuda.manual_seed(worker_seed)
            torch.cuda.manual_seed_all(worker_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.gpu_id = gpu_id
        # self.device = torch.device(f"cuda:{gpu_id}")
        self.device = torch.device("cuda")

        import logging
        worker_logger = logging.getLogger(__name__)
        worker_logger.info(f"Worker {gpu_id}: 初始化模型...")

        # 加载配置
        config = OmegaConf.load(config_path)

        # 构建模型
        self.model = construct_class_by_name(**config.Model.kwargs, logger=worker_logger)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        self.model = self.model.to(self.device)

        self.tokenizer = self.model.tokenizer

        logger.info(f"Worker {gpu_id}: 模型加载完成")

    def process_batch(self, batch_data: List[Tuple[int, Dict[str, Any]]]) -> List[Tuple[int, str]]:
        """
        处理一个batch的数据

        Args:
            batch_data: List of (sample_idx, sample_dict)

        Returns:
            List of (sample_idx, generated_sequence)
        """
        if not batch_data:
            return []

        indices = [item[0] for item in batch_data]
        samples = [item[1] for item in batch_data]

        # 准备batch
        all_domains = []
        num_domains_per_protein = []

        for sample in samples:
            domains = sample["domain"]
            all_domains.extend(domains)
            num_domains_per_protein.append(len(domains))

        # Tokenize
        encodings = self.tokenizer(
            all_domains,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="longest"
        )

        domain_ids = encodings.input_ids.to(self.device)
        domain_masks = encodings.attention_mask.to(self.device)
        num_domains_per_protein = torch.tensor(num_domains_per_protein).to(self.device)

        # 生成
        with torch.no_grad():
            output = self.model.generate(
                domain_ids=domain_ids,
                domain_masks=domain_masks,
                num_domains_per_protein=num_domains_per_protein
            )

        generated_seqs = output["output_seqs"]

        # 返回结果
        results = list(zip(indices, generated_seqs))
        return results




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/DomainComb/configs/02-AR-esm2-qwen3-1.2B.yaml")
    parser.add_argument("--tsv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb/large_scale_design/results")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # 设置随机种子（如果提供）
    if args.seed is not None:
        set_seed(args.seed)

    # 初始化Ray
    logger.info("初始化Ray...")
    # 如果在Slurm多节点环境中，连接到已启动的Ray集群
    # 否则在本地启动Ray
    if "SLURM_JOB_ID" in os.environ and int(os.environ.get("SLURM_NNODES", "1")) > 1:
        logger.info("检测到多节点Slurm环境，连接到Ray集群...")
        ray.init(address='auto')
    else:
        logger.info("单节点环境，本地启动Ray...")
        ray.init(num_gpus=args.num_gpus)

    # 加载数据集
    logger.info("加载数据集...")
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    dataset = Step2Dataset(tsv_path=args.tsv_path, tokenizer=tokenizer)
    logger.info(f"数据集大小: {len(dataset)}")

    # 检查断点续传：读取已完成的索引
    completed_indices = set()
    output_path = os.path.join(args.output_dir, os.path.basename(args.tsv_path).replace(".tsv", ".jsonl"))
    if os.path.exists(output_path):
        logger.info(f"检测到已存在的输出文件，加载已完成的索引...")
        with open(output_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    completed_indices.add(data["index"])
                except:
                    continue
        logger.info(f"已完成 {len(completed_indices)} 个样本，将跳过这些样本")

    # 按长度排序并分batch
    all_batches = sort_and_batch_by_length(dataset, args.batch_size)

    # 过滤掉已完成的batch（如果batch中所有样本都已完成）
    filtered_batches = []
    for batch in all_batches:
        # 过滤掉已完成的样本
        remaining_samples = [(idx, sample) for idx, sample in batch if idx not in completed_indices]
        if remaining_samples:
            filtered_batches.append(remaining_samples)

    logger.info(f"总共 {len(all_batches)} 个batches，跳过已完成的后剩余 {len(filtered_batches)} 个batches")

    if not filtered_batches:
        logger.info("所有样本已完成，无需继续处理")
        ray.shutdown()
        return

    # 加载配置获取checkpoint路径
    config = OmegaConf.load(args.config)
    checkpoint_path = config.Checkpoint_path

    # 创建workers
    logger.info(f"创建 {args.num_gpus} 个workers...")
    workers = [
        InferenceWorker.remote(args.config, checkpoint_path, gpu_id, args.seed)
        for gpu_id in range(args.num_gpus)
    ]

    # 提交所有任务，让Ray自动调度
    logger.info(f"提交 {len(filtered_batches)} 个任务到 {args.num_gpus} 个workers...")
    pending_tasks = []
    for i, batch in enumerate(filtered_batches):
        # 轮询分配给workers，Ray会自动排队管理
        worker = workers[i % args.num_gpus]
        task = worker.process_batch.remote(batch)
        pending_tasks.append(task)

    # 打开文件用于追加写入
    output_file = open(output_path, "a")
    total_generated = len(completed_indices)

    # 等待所有任务完成，边生成边写入
    try:
        with tqdm(total=len(filtered_batches), desc="处理进度") as pbar:
            while pending_tasks:
                # 等待任意一个任务完成
                done_tasks, pending_tasks = ray.wait(pending_tasks, num_returns=1)

                # 获取完成的结果并立即写入文件
                for task in done_tasks:
                    batch_results = ray.get(task)
                    for idx, seq in batch_results:
                        original_sample = dataset[idx]
                        output_data = {
                            "index": idx,
                            "input_domains": original_sample["domain"],
                            "generated_sequence": seq
                        }
                        output_file.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                        output_file.flush()  # 立即刷新到磁盘
                        total_generated += 1
                    pbar.update(1)
    finally:
        output_file.close()

    logger.info(f"完成！共生成 {total_generated} 条序列（包含之前已完成的 {len(completed_indices)} 条）")

    # 关闭Ray
    ray.shutdown()


if __name__ == "__main__":
    main()
