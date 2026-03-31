source ~/miniconda3/etc/profile.d/conda.sh
conda activate foldflow-env
cd ~/fengyuan/Pretrain
test_output_path=$1
test_output_path=$(readlink -f "$test_output_path")

cd /storage/yuanfajieLab/yuanfajie/my_project/analysis/tmscore_foldseek/workdir
source ~/miniconda3/etc/profile.d/conda.sh
conda activate foldseek
start_time=$(date +%s)

# Create temporary directory for main model.cif files only
TEMP_QUERY_DIR=$test_output_path/af3_output_main_models
mkdir -p $TEMP_QUERY_DIR

echo "Creating symbolic links for main model.cif files..."
# Find all main model.cif files (not in seed-* subdirectories) and create symlinks
find $test_output_path/af3_output -maxdepth 2 -name "*_model.cif" -not -path "*/seed-*/*" -type f | while read cif_file; do
    ln -sf "$cif_file" "$TEMP_QUERY_DIR/$(basename $cif_file)"
done

num_files=$(ls -1 $TEMP_QUERY_DIR | wc -l)
echo "Found $num_files main model.cif files"
QUERY_DIR=$TEMP_QUERY_DIR
OUTPUT_DIR=$test_output_path
OUTPUT_BASENAME=af3_output
echo "QUERY_DIR: $QUERY_DIR"
echo "OUTPUT_FILE: ${OUTPUT_DIR}/${OUTPUT_BASENAME}_vs_pdb.txt"
echo "Running foldseek easy-search..."
foldseek easy-search $QUERY_DIR pdb ${OUTPUT_DIR}/${OUTPUT_BASENAME}_vs_pdb.txt tmpFolder\
     --alignment-type 1\
     --format-output query,target,qtmscore,ttmscore,alntmscore,lddt\
     --tmscore-threshold 0.0\
     --exhaustive-search\
     --max-seqs 10000000000\
     --threads 120

# Clean up temporary directory
# echo "Cleaning up temporary directory..."
# rm -rf $TEMP_QUERY_DIR

end_time=$(date +%s)

duration=$((end_time - start_time))
echo "Elapsed time: $duration s"
