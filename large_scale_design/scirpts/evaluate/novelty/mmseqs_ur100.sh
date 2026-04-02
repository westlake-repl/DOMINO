cd /storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb
MMSEQS="/storage/yuanfajieLab/yuanfajie/my_project/WetExperiments/mmseqs_scripts/mmseqs/bin/mmseqs"

start_time=$(date +%s)

QUERY_FASTA_PATH=$1
QUERY_DIR=$(dirname $QUERY_FASTA_PATH)

UR100_DB_PATH=/storage/yuanfajieLab/yuanfajie/my_project/WetExperiments/mmseqs_scripts/database/UniRef100_gpu
WORKING_PATH=/storage/yuanfajieLab/yuanfajie/my_project/WetExperiments/mmseqs_scripts

## random id
random_id=$(date +%s%N)
TMP_DIR=$WORKING_PATH/tmp_$(basename $QUERY_DIR)_$random_id
RESULT_DB=$WORKING_PATH/resultDB/RESULT_DB_$(basename $QUERY_DIR)_$random_id
QUERY_DB=$WORKING_PATH/queryDB/QUERY_DB_$(basename $QUERY_DIR)_$random_id
alnNew=$WORKING_PATH/alnNew/alnNew_$(basename $QUERY_DIR)_$random_id

ALIGN_FASTA_PATH="${QUERY_FASTA_PATH%.fasta}_mmseqs_vs_ur100.tsv"

echo $ALIGN_FASTA_PATH
if [ -d $TMP_DIR ]; then
    echo "TMP_DIR exists, remove it"
    echo "Be careful, this will remove all files in $TMP_DIR"
    # read -p "Press any key to continue, or press Ctrl+C to exit"
    rm -rf $TMP_DIR
fi
mkdir -p $TMP_DIR

# cd $DB_DIR
# # create query DB
$MMSEQS createdb $QUERY_FASTA_PATH $QUERY_DB
$MMSEQS search $QUERY_DB $UR100_DB_PATH $RESULT_DB $TMP_DIR --gpu 1 --min-seq-id 0 --alignment-mode 3 --max-seqs 100 -s 1 -c 0.8 --cov-mode 0 --threads 120 -a
$MMSEQS align $QUERY_DB $UR100_DB_PATH $RESULT_DB $alnNew -a --threads 120
$MMSEQS convertalis $QUERY_DB $UR100_DB_PATH $alnNew $ALIGN_FASTA_PATH --format-output query,target,qseq,tseq,pident,fident,nident --threads 120

end_time=$(date +%s)
echo "Time taken: $((end_time - start_time)) seconds"