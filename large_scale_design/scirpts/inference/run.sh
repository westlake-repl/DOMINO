cd /storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb
source ~/miniconda3/etc/profile.d/conda.sh
conda activate foldflow-env

cd /storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb
python large_scale_design/scirpts/inference/run_ray.py --tsv_path $1 --seed 0