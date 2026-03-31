
cd /storage/yuanfajieLab/yuanfajie/my_project/analysis/tmscore_foldseek/workdir
source ~/miniconda3/etc/profile.d/conda.sh
conda activate foldseek

test_output_path=$1
test_output_path=$(readlink -f "$test_output_path")

TEMP_QUERY_DIR=$test_output_path/af3_output_main_models
RES_DIR=$test_output_path/af3_output_main_models_cluster_res
mkdir $RES_DIR
foldseek easy-cluster $TEMP_QUERY_DIR $RES_DIR/res tmpFolder \
    --cov-mode 0 \
    --alignment-type 1\
    --min-seq-id 0\
    --tmscore-threshold 0.5 \
    --threads 64
