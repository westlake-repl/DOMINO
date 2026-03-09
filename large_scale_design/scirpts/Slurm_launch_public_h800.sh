#!/bin/bash

#SBATCH -p public-h800
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 120
#SBATCH --mem 1800G
#SBATCH -J TED
#SBATCH -o /storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/output/SlurmLogs/%j.log
#SBATCH --gres=gpu:8

bash "$@" 