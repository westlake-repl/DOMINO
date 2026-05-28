# python large_scale_design/process_from_retrieval_to_input/00-tmscore_calculation.py \
#     --input_path /storage/yuanfajieLab/yuanfajie/sujin/Datasets/TED/embedding/afdb_cluster_power0.75/retrieval_results.tsv \
#     --prefetched_file large_scale_design/results/LargeScale-02-0324-retrieval_results/LargeScale-02-0324-retrieval_results_random_5000.tsv \
#     --output_path large_scale_design/results/LargeScale-02-0324-retrieval_results/LargeScale-02-0324-retrieval_results_random_5000_with_tmscore.tsv

  python large_scale_design/process_from_retrieval_to_input/01-get_input_tmscore.py \
    --input_path large_scale_design/results/LargeScale-02-0324-retrieval_results/LargeScale-02-0324-retrieval_results_random_5000_with_tmscore.tsv \
    --prefetched_file large_scale_design/results/LargeScale-02-0324-retrieval_results/LargeScale-02-0324-retrieval_results_random_5000.tsv \
    --output_name LargeScale-02-0324-retrieval_results_random_5000_tmscore_lt_0.5.tsv