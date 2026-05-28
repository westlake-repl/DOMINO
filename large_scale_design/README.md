## Notes

/storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb/large_scale_design/inputs/LargeScale-02-0324-retrieval_results_random5000_backup.tsv 是/storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb/large_scale_design/inputs/LargeScale-02-0324-retrieval_results.tsv 中随机才到的5000的 query的backup文件

/storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb/large_scale_design/scirpts/prepare_inputs/01-get_input_seqid.py.old 原本是给这个sujin/Datasets/TED/embedding/afdb_cluster_power0.75/retrieval_results.tsv下面的tsv 然后算seq id的。现在改成01-get_input_seqid.py，直接根据LargeScale-02-0324-retrieval_results_random5000_backup.tsv算seq id

类似的改动也有01-get_input_tmscore.py.old