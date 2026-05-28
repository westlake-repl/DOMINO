import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import biotite.structure.io as bsio
import lmdb
import numpy as np
from tqdm import tqdm


TMALIGN_EXEC = "/storage/yuanfajieLab/yuanfajie/my_project/analysis/structural_comparison/TMscore/TMalign"
SEQ_LMDB_PATH = "/storage/yuanfajieLab/yuanfajie/datasets/AFDB/LMDB_seqonly"
DOMAIN_LMDB_PATH = "/storage/yuanfajieLab/yuanfajie/datasets/AFDB/LMDB_uid2domain_info"
AF_STRUCTURE_DIR = "/storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb/.cache/af_structures"
DOMAIN_STRUCTURES_DIR = "/storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb/.cache/domain_structures"
INPUT_PATH = "/storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb/large_scale_design/inputs/LargeScale-02-0324-retrieval_results_random5000_backup.tsv"
OUTPUT_PATH = "/storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb/large_scale_design/inputs/LargeScale-02-0324-retrieval_results_random_5000_tmscore_lt_0.5.tsv"


seq_env = None
domain_env = None
seq_txn = None
domain_txn = None


def compute_domain_hash(domain_seq):
    return hashlib.md5(domain_seq.encode()).hexdigest()


def sa2aa(sa):
    return "<unk>".join([piece[::2] for piece in sa.split("<unk>")])


def remove_unk(seq):
    return seq.replace("<unk>", "")


def load_json(path):
    if path is None or not Path(path).exists():
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def dump_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def get_column_idx(header, name, default_idx):
    if name in header:
        return header.index(name)
    return default_idx


def get_output_path(output_name):
    output_path = Path(output_name)
    if output_path.is_absolute():
        return output_path
    return Path("/storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb/large_scale_design/inputs") / output_path


def init_lmdb(args):
    global seq_env, domain_env, seq_txn, domain_txn

    seq_env = lmdb.open(args.seq_lmdb_path, readonly=True, lock=False)
    domain_env = lmdb.open(args.domain_lmdb_path, readonly=True, lock=False)
    seq_txn = seq_env.begin()
    domain_txn = domain_env.begin()


def name2seq(name):
    seq = seq_txn.get(name.encode("utf-8"))
    if seq is None:
        raise KeyError(f"sequence is missing in LMDB: {name}")
    return seq.decode("utf-8")


def name2domain_info_list(name):
    domain_info = domain_txn.get(name.encode("utf-8"))
    if domain_info is None:
        raise KeyError(f"domain info is missing in LMDB: {name}")
    return json.loads(domain_info.decode("utf-8"))


def domain_fillin(sequence, domain_info):
    domain_info = domain_info.split("_")
    domain_info = [tuple(map(int, domain.split("-"))) for domain in domain_info]
    domain_info = sorted(domain_info, key=lambda x: x[0])
    domain_filled_sequence = []
    for domain in domain_info:
        start, end = domain
        domain_filled_sequence.append(sequence[start - 1 : end])
    return "".join(domain_filled_sequence)


def domain_seq_to_domain_info(domain_seq, gt_seq):
    pieces = domain_seq.split("<unk>")
    if not pieces:
        return None

    positions = []
    search_start = 0
    for piece in pieces:
        pos = gt_seq.find(piece, search_start)
        if pos == -1:
            return None
        start = pos + 1
        end = pos + len(piece)
        positions.append(f"{start}-{end}")
        search_start = pos + len(piece)

    return "_".join(positions)


def decompress_gz_file(gz_path, output_path):
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def ensure_af_structure(uniprot_id, args):
    local_dir = Path(args.af_structure_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    cif_filename = f"AF-{uniprot_id}-F1-model_v4.cif"
    gz_filename = f"{cif_filename}.gz"
    local_cif_path = local_dir / cif_filename
    local_gz_path = local_dir / gz_filename

    if local_cif_path.exists() and local_cif_path.stat().st_size > 0:
        return str(local_cif_path)

    remote_path = f"{args.remote_host}:{args.remote_base_path}/{gz_filename}"
    scp_cmd = ["scp", "-q", remote_path, str(local_dir)]
    result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=args.download_timeout)
    if result.returncode != 0:
        raise RuntimeError(f"download failed for {uniprot_id}: {result.stderr.strip()}")

    if not local_gz_path.exists() or local_gz_path.stat().st_size == 0:
        raise RuntimeError(f"downloaded gz file is missing or empty: {local_gz_path}")

    decompress_gz_file(local_gz_path, local_cif_path)
    os.remove(local_gz_path)

    if not local_cif_path.exists() or local_cif_path.stat().st_size == 0:
        raise RuntimeError(f"decompressed cif file is missing or empty: {local_cif_path}")

    return str(local_cif_path)


def extract_domain_structure(gt_structure_path, domain_info, output_pdb_path):
    structure = bsio.load_structure(gt_structure_path, model=1)

    all_atoms = []
    for pos_range in domain_info.split("_"):
        start, end = map(int, pos_range.split("-"))
        mask = (structure.res_id >= start) & (structure.res_id <= end)
        all_atoms.append(structure[mask])

    if len(all_atoms) == 1:
        combined_structure = all_atoms[0]
    else:
        combined_structure = all_atoms[0]
        for fragment in all_atoms[1:]:
            combined_structure = combined_structure + fragment

    unique_res_ids = np.unique(combined_structure.res_id)
    res_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_res_ids, start=1)}
    combined_structure.res_id = np.array([res_id_mapping[old_id] for old_id in combined_structure.res_id])
    bsio.save_structure(output_pdb_path, combined_structure)


def get_domain_structure(uniprot_id, domain_seq, domain_info, args):
    domain_hash = compute_domain_hash(domain_seq)
    structures_dir = Path(args.domain_structures_dir) / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)
    domain_pdb_path = structures_dir / f"{domain_hash}.pdb"

    if domain_pdb_path.exists() and domain_pdb_path.stat().st_size > 0:
        return {
            "domain_hash": domain_hash,
            "domain_seq": domain_seq,
            "domain_pdb_path": str(domain_pdb_path),
        }

    gt_structure_path = ensure_af_structure(uniprot_id, args)
    extract_domain_structure(gt_structure_path, domain_info, str(domain_pdb_path))

    return {
        "domain_hash": domain_hash,
        "domain_seq": domain_seq,
        "domain_pdb_path": str(domain_pdb_path),
    }


def get_candidate_domain_structure(candidate_seg, candidate_uniprot_ids, args):
    candidate_domain_seq = sa2aa(candidate_seg)
    for uniprot_id in candidate_uniprot_ids:
        gt_seq = name2seq(uniprot_id)
        domain_info = domain_seq_to_domain_info(candidate_domain_seq, gt_seq)
        if domain_info is None:
            continue
        return get_domain_structure(uniprot_id, candidate_domain_seq, domain_info, args)

    raise RuntimeError(
        "candidate domain position is not found in any candidate uid: "
        f"{candidate_uniprot_ids}; candidate={candidate_domain_seq[:80]}"
    )


def get_gt_target_domain_structures(gt_uniprot_ids, query_seq_aa, args):
    gt_target_domains = []
    for uniprot_id in gt_uniprot_ids:
        gt_seq = name2seq(uniprot_id)
        domain_info_list = name2domain_info_list(uniprot_id)
        for domain_info in domain_info_list:
            domain_seq = domain_fillin(gt_seq, domain_info)
            if domain_seq == query_seq_aa:
                continue
            gt_target_domains.append(get_domain_structure(uniprot_id, domain_seq, domain_info, args))

    if not gt_target_domains:
        raise RuntimeError(f"no GT target domains are available after removing query domain: {gt_uniprot_ids}")

    return gt_target_domains


def parse_tmalign_output(output_path):
    scores = []
    with open(output_path, 'r') as f:
        for line in f:
            if line.startswith("TM-score="):
                fields = line.replace("=", " ").split()
                if len(fields) >= 2:
                    scores.append(float(fields[1]))
            if len(scores) >= 2:
                break
    if not scores:
        raise RuntimeError(f"no TM-score found in TMalign output: {output_path}")
    return max(scores)


def is_valid_tmscore(tm_score):
    try:
        return math.isfinite(float(tm_score))
    except (TypeError, ValueError):
        return False


def make_pair_key(hash1, hash2):
    first, second = sorted([hash1, hash2])
    return f"{first}\t{second}"


def calculate_tmscore(candidate_domain, gt_target_domain, args, tmscore_cache):
    key = make_pair_key(candidate_domain["domain_hash"], gt_target_domain["domain_hash"])
    if key in tmscore_cache and is_valid_tmscore(tmscore_cache[key]):
        return float(tmscore_cache[key]), False

    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)
    fd, outpath = tempfile.mkstemp(prefix="tmp_tmalign_", suffix=".txt", dir=args.tmp_dir)
    os.close(fd)

    try:
        with open(outpath, 'w') as f:
            result = subprocess.run(
                [args.tmalign_exec, candidate_domain["domain_pdb_path"], gt_target_domain["domain_pdb_path"]],
                stdout=f,
                stderr=subprocess.DEVNULL,
                timeout=args.timeout,
            )
        if result.returncode != 0:
            raise RuntimeError(
                "TMalign failed: "
                f"{candidate_domain['domain_pdb_path']} vs {gt_target_domain['domain_pdb_path']}"
            )
        tm_score = parse_tmalign_output(outpath)
        tmscore_cache[key] = tm_score
        return float(tm_score), True
    finally:
        if os.path.exists(outpath):
            os.remove(outpath)


def calculate_max_tmscore(candidate_domain, gt_target_domains, args, tmscore_cache):
    scores = []
    cache_changed = False
    for gt_target_domain in gt_target_domains:
        tm_score, changed = calculate_tmscore(candidate_domain, gt_target_domain, args, tmscore_cache)
        scores.append(tm_score)
        cache_changed = cache_changed or changed
    return max(scores), cache_changed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=INPUT_PATH)
    parser.add_argument("--output_name", type=str, default=OUTPUT_PATH)
    parser.add_argument("--tmscore_threshold", type=float, default=0.5)
    parser.add_argument("--af_structure_dir", type=str, default=AF_STRUCTURE_DIR)
    parser.add_argument("--domain_structures_dir", type=str, default=DOMAIN_STRUCTURES_DIR)
    parser.add_argument("--seq_lmdb_path", type=str, default=SEQ_LMDB_PATH)
    parser.add_argument("--domain_lmdb_path", type=str, default=DOMAIN_LMDB_PATH)
    parser.add_argument("--tmalign_exec", type=str, default=TMALIGN_EXEC)
    parser.add_argument("--tmp_dir", type=str, default="/storage/yuanfajieLab/yuanfajie/tmpfile")
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--remote_host", type=str, default="TempCluster_yungu")
    parser.add_argument("--remote_base_path", type=str, default="/ssd/share-data/gzfile")
    parser.add_argument("--download_workers", type=int, default=8)
    parser.add_argument("--download_timeout", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max_queries", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    init_lmdb(args)

    output_path = get_output_path(args.output_name)
    if args.cache_path is None:
        cache_path = str(output_path.with_suffix(".tmscore_cache.json"))
    else:
        cache_path = args.cache_path
    tmscore_cache = load_json(cache_path)

    with open(args.input_path, 'r') as f:
        lines = f.readlines()

    header_line = lines[0]
    header = header_line.strip().split("\t")
    query_idx = get_column_idx(header, "Query_Segment", 0)
    candidate_idx = get_column_idx(header, "Candidate_Segment", 1)
    gt_idx = get_column_idx(header, "Ground_Truth_Uniprot_IDs", 5)
    candidate_uid_idx = get_column_idx(header, "Candidate_Uniprot_IDs", len(header) - 1)

    if "Max_TMscore" in header:
        tmscore_idx = header.index("Max_TMscore")
        output_header = header_line
    else:
        tmscore_idx = len(header)
        output_header = header_line.rstrip("\n") + "\tMax_TMscore\n"

    query_seg_to_lines = defaultdict(list)
    for line in tqdm(lines[1:], desc="Reading input rows"):
        line_list = line.strip().split("\t")
        if len(line_list) <= max(query_idx, candidate_idx, gt_idx, candidate_uid_idx):
            raise ValueError(f"malformed input line: {line[:200]}")
        query_seg_to_lines[line_list[query_idx]].append(line)

    query_segs = list(query_seg_to_lines.keys())
    if args.max_queries is not None:
        query_segs = query_segs[:args.max_queries]

    output_rows = [output_header]
    selected_num = 0
    cache_changed = False
    calculated_scores = 0

    for query_seg in tqdm(query_segs, desc="Selecting by TMscore"):
        query_seq_aa = remove_unk(sa2aa(query_seg))
        query_selected = False

        for line in query_seg_to_lines[query_seg]:
            line_list = line.rstrip("\n").split("\t")
            candidate_seg = line_list[candidate_idx]
            gt_uniprot_ids = [uid.strip() for uid in line_list[gt_idx].split(",") if uid.strip()]
            candidate_uniprot_ids = [uid.strip() for uid in line_list[candidate_uid_idx].split(",") if uid.strip()]

            candidate_domain = get_candidate_domain_structure(candidate_seg, candidate_uniprot_ids, args)
            gt_target_domains = get_gt_target_domain_structures(gt_uniprot_ids, query_seq_aa, args)
            tm_score, changed = calculate_max_tmscore(candidate_domain, gt_target_domains, args, tmscore_cache)
            cache_changed = cache_changed or changed
            calculated_scores += 1

            if tm_score < args.tmscore_threshold:
                tm_score_str = f"{tm_score:.6f}"
                if tmscore_idx < len(line_list):
                    line_list[tmscore_idx] = tm_score_str
                else:
                    line_list.append(tm_score_str)
                output_rows.append("\t".join(line_list) + "\n")
                selected_num += 1
                query_selected = True
                break

            if cache_changed and calculated_scores % 100 == 0:
                dump_json(cache_path, tmscore_cache)
                cache_changed = False

        if not query_selected:
            print(f"no candidate with TMscore < {args.tmscore_threshold}: {query_seg[:80]}")

    if cache_changed:
        dump_json(cache_path, tmscore_cache)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for line in output_rows:
            f.write(line)

    print(f"processed queries: {len(query_segs)}")
    print(f"selected queries: {selected_num}")
    print(f"writing into {output_path}")


if __name__ == "__main__":
    main()
