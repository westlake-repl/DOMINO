import argparse
from pathlib import Path
import subprocess
from tqdm import tqdm
import numpy as np
import json
import biotite.structure.io as bsio
import biotite.sequence.io as bseqio
from multiprocessing import Pool
import random 

tmalign_exec = "/storage/yuanfajieLab/yuanfajie/my_project/analysis/structural_comparison/TMscore/TMalign"
GT_path = "/storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/analysis/AFDB_structure_download/0327-af_structures"

def GTTemplateName(UniID):
    # path = f"{GT_path}/AF-{UniID}-F1-model_v4.pdb"
    path = f"{GT_path}/{UniID}.pdb"

    # if not os.path.exists(path):
    #     print(f"{path} is not exists.")
    return path

def calculatetmscore(pdb1, pdb2):
    # now we only return query tmscore
    random_id = random.randint(1, 1000000)

    outpath = f"/storage/yuanfajieLab/yuanfajie/tmpfile/tmp_{random_id}.txt"

    tmalign_cmd = f"{tmalign_exec} {pdb1} {pdb2}> {outpath}"
    subprocess.call(tmalign_cmd, shell=True)
    # read output file and get tmscore
    content = open(outpath, 'r').readlines()
    query_line = content[13]
    target_line = content[14]
    q_tm_score = float(query_line.split()[1])
    t_tm_score = float(target_line.split()[1])
    # return max(q_tm_score, t_tm_score)
    return q_tm_score

def read_seq_from_pdb(file_path):
    aa_codes = {
        'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E',
        'PHE':'F', 'GLY':'G', 'HIS':'H', 'LYS':'K',
        'ILE':'I', 'LEU':'L', 'MET':'M', 'ASN':'N',
        'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S',
        'THR':'T', 'VAL':'V', 'TYR':'Y', 'TRP':'W'}

    seq = ''
    
    for line in open(file_path):
        if line[0:4] == "ATOM":
            columns = line.split()
            if columns[2] == "CA":
                seq = seq + aa_codes[columns[3]]
    return seq

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--output_structure_type", type=str, default="af3_output", choices=["esmfold_results", "af3_output"])
    parser.add_argument("--generation_path", type=str, default="/storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb/large_scale_design/inputs/LargeScale-01-0301-retrieval_results.tsv")
    parser.add_argument("--num_processes", type=int, default=32, help="Number of processes for parallel computation")
    args = parser.parse_args()
    print("Attention!!! We use max(qTM,tTM) to calculate the structure similarity.")

    target_path = Path(args.jsonl_path).parent
    generation_path = args.generation_path

    all_generation_content = open(generation_path, 'r').readlines()

    if args.output_structure_type == "esmfold_results":
        raise NotImplementedError("We have not implemented the TMscore calculation for esmfold results yet.")
        designed_structure_path = list((Path(target_path, args.output_structure_type)).glob("*.pdb"))
        print("find number of structure paths:", len(designed_structure_path))
        
        # designed_structure_path = sorted(designed_structure_path, key=lambda x: int(str(x).split("_")[-1].split(".")[0]))
    elif args.output_structure_type == "af3_output":
        designed_structure_path = list(Path(target_path, args.output_structure_type).glob("*/*.cif"))
        print("find number of structure paths:", len(designed_structure_path))
        idx2structure_path = { int(str(path.name).split(".")[0].replace("_model", "")): path for path in designed_structure_path}
        ## the idx is also the index of all_generation_content
        # designed_structure_path = sorted(designed_structure_path, key=lambda x: int(str(x).split("_")[-2]))
    
    # print(idx2structure_path)
    # print(idx2structure_path[4121248])
    # print(all_generation_content[4121248])
    # print(all_generation_content[4121247])

    # print(all_generation_content[4121249])

    # assert 0
    def process_single_pair(idx):
        target_line = all_generation_content[idx + 1].strip().split("\t")
        query_uid = target_line[-3].split(",")
        target_uid = target_line[-1].split(",")
        query_pdb_path = [Path(GT_path)/f"AF-{q_uid}-F1-model_v4.cif" for q_uid in query_uid]
        target_pdb_path = [Path(GT_path)/f"AF-{t_uid}-F1-model_v4.cif" for t_uid in target_uid]
        TMscore_query_list = []
        TMscore_target_list = []
        for q_pdb_path in query_pdb_path:
            TMscore_query_list.append(calculatetmscore(idx2structure_path[idx], q_pdb_path))
        for t_pdb_path in target_pdb_path:
            TMscore_target_list.append(calculatetmscore(idx2structure_path[idx], t_pdb_path))
        
        TMscore_query = np.max(TMscore_query_list)
        TMscore_target = np.max(TMscore_target_list)
        return TMscore_query, TMscore_target

    # tasks = [(idx, query_uid, target_uid) for idx, (query_uid, target_uid) in enumerate(uid_list)]
    tasks = list(idx2structure_path.keys())

    with Pool(processes=args.num_processes) as pool:
        results = list(tqdm(pool.imap(process_single_pair, tasks), total=len(tasks), desc="calculating TMscore"))

    TMscore_query_structure_dict = {idx: r[0] for idx, r in zip(tasks, results)}
    TMscore_target_structure_dict = {idx: r[1] for idx, r in zip(tasks, results)}
    print(f"Mean_TMscore_query_structure_{args.output_structure_type}", np.mean(list(TMscore_query_structure_dict.values())))
    print(f"Mean_TMscore_target_structure_{args.output_structure_type}", np.mean(list(TMscore_target_structure_dict.values())))

    metrics = json.load(open(Path(target_path) / "log_metrics.json", 'r'))
    metrics[f"Mean_TMscore_query_structure_{args.output_structure_type}"] = np.mean(list(TMscore_query_structure_dict.values()))
    metrics[f"Mean_TMscore_target_structure_{args.output_structure_type}"] = np.mean(list(TMscore_target_structure_dict.values()))
    metrics[f"TMscore_query_structure_{args.output_structure_type}"] = TMscore_query_structure_dict
    metrics[f"TMscore_target_structure_{args.output_structure_type}"] = TMscore_target_structure_dict
    with open(Path(target_path) / "log_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)