#!/bin/bash

cd /storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb
source ~/miniconda3/etc/profile.d/conda.sh
conda activate foldflow-env

# 获取节点列表
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# 获取一个空闲端口
port=6379

# 设置Ray临时目录
export RAY_TMPDIR=/tmp/ray_${SLURM_JOB_ID}

echo "启动Ray集群..."
echo "Head节点: $head_node ($head_node_ip:$port)"

# 在head节点启动Ray head
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &

# 等待head节点启动
sleep 10

# 在其他worker节点启动Ray worker
for ((i = 1; i < ${#nodes_array[@]}; i++)); do
    node_i=${nodes_array[$i]}
    echo "启动worker节点: $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address="$head_node_ip:$port" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &
    sleep 5
done

# 等待所有Ray节点启动完成
sleep 10

# 运行Python脚本（只在head节点运行）
echo "运行推理任务..."
python -u large_scale_design/scirpts/run_ray.py \
    --tsv_path $1 \
    --seed 0 \
    --num_gpus 16

# 关闭Ray集群
ray stop
