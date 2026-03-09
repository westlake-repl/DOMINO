#!/bin/bash

#SBATCH -p public-h800
#SBATCH -N 4
#SBATCH -n 4
#SBATCH -c 120
#SBATCH --mem 1800G
#SBATCH -J TED_3nodes
#SBATCH -o /storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/output/SlurmLogs/%j.log
#SBATCH --gres=gpu:8

bash "$@"
