import argparse
import gzip
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path

import biotite.structure.io as bsio
import lmdb
import numpy as np
from tqdm import tqdm


TMALIGN_EXEC = "/storage/yuanfajieLab/yuanfajie/my_project/analysis/structural_comparison/TMscore/TMalign"
SEQ_LMDB_PATH = "/storage/yuanfajieLab/yuanfajie/datasets/AFDB/LMDB_seqonly"
AF_STRUCTURE_DIR = "/storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/analysis/AFDB_structure_download/0403-af_structures"
DOMAIN_STRUCTURES_DIR = "/storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/analysis/AFDB_structure_download/retrieval_domain_structures"


def compute_domain_hash(domain_seq):
    return hashlib.md5(domain_seq.encode()).hexdigest()


def normalize_domain_seq(domain_seq):
    return "<unk>".join([piece[::2] for piece in domain_seq.split("<unk>")])


def load_prefetched_query_segs(prefetched_file):
    with open(prefetched_file, 'r') as f:
        return set(line.strip().split("\t")[0] for line in f.readlines()[1:] if line.strip())


def get_column_idx(header, name, default_idx):
    if name in header:
        return header.index(name)
    return default_idx


def parse_retrieval_rows(input_path, prefetched_file):
    query_segs = load_prefetched_query_segs(prefetched_file)
    domain_to_uniprots = defaultdict(list)
    rows = []

    with open(input_path, 'r') as f:
        header_line = f.readline()
        header = header_line.strip().split("\t")

        query_idx = get_column_idx(header, "Query_Segment", 0)
        candidate_idx = get_column_idx(header, "Candidate_Segment", 1)
        gt_idx = get_column_idx(header, "Ground_Truth_Uniprot_IDs", 5)
        candidate_uid_idx = get_column_idx(header, "Candidate_Uniprot_IDs", len(header) - 1)

        for line in tqdm(f, desc="Reading retrieval rows"):
            line_list = line.strip().split("\t")
            if len(line_list) <= max(query_idx, candidate_idx, gt_idx, candidate_uid_idx):
                continue

            query_seg = line_list[query_idx]
            if query_seg not in query_segs:
                continue

            candidate_seg = line_list[candidate_idx]
            query_domain = normalize_domain_seq(query_seg)
            candidate_domain = normalize_domain_seq(candidate_seg)
            gt_uid = line_list[gt_idx].split(",")[0].strip()
            candidate_uid = line_list[candidate_uid_idx].split(",")[0].strip()

            if gt_uid and gt_uid not in domain_to_uniprots[query_domain]:
                domain_to_uniprots[query_domain].append(gt_uid)
            if candidate_uid and candidate_uid not in domain_to_uniprots[candidate_domain]:
                domain_to_uniprots[candidate_domain].append(candidate_uid)

            rows.append({
                "line": line,
                "query_domain": query_domain,
                "candidate_domain": candidate_domain,
                "query_hash": compute_domain_hash(query_domain),
                "candidate_hash": compute_domain_hash(candidate_domain),
            })

    return header_line, header, rows, domain_to_uniprots


def file_signature(path):
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path),
        "mtime": stat.st_mtime,
        "size": stat.st_size,
    }


def get_parse_cache_path(args):
    if args.parse_cache_path is not None:
        return args.parse_cache_path
    return str(Path(args.output_path).with_suffix(".retrieval_rows_cache.json"))


def load_parse_cache(args):
    cache_path = Path(get_parse_cache_path(args))
    if args.force_reparse or not cache_path.exists():
        return None

    cache = load_json(cache_path)
    meta = cache.get("meta", {})
    if meta.get("input_path") != file_signature(args.input_path):
        return None
    if meta.get("prefetched_file") != file_signature(args.prefetched_file):
        return None

    domain_to_uniprots = defaultdict(list)
    for domain_seq, uniprot_ids in cache["domain_to_uniprots"].items():
        domain_to_uniprots[domain_seq].extend(uniprot_ids)

    print(f"loaded retrieval rows cache from {cache_path}")
    return cache["header_line"], cache["header"], cache["rows"], domain_to_uniprots


def save_parse_cache(args, header_line, header, rows, domain_to_uniprots):
    cache_path = Path(get_parse_cache_path(args))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        "meta": {
            "input_path": file_signature(args.input_path),
            "prefetched_file": file_signature(args.prefetched_file),
        },
        "header_line": header_line,
        "header": header,
        "rows": rows,
        "domain_to_uniprots": dict(domain_to_uniprots),
    }
    dump_json(cache_path, cache)
    print(f"retrieval rows cache saved to {cache_path}")


def load_or_parse_retrieval_rows(args):
    cached_data = load_parse_cache(args)
    if cached_data is not None:
        return cached_data

    parsed_data = parse_retrieval_rows(args.input_path, args.prefetched_file)
    save_parse_cache(args, *parsed_data)
    return parsed_data


def load_json(path):
    if not Path(path).exists():
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def dump_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def read_gt_sequence(seq_lmdb_path, entry_id):
    env = lmdb.open(seq_lmdb_path, lock=False, readonly=True, map_size=1024**4)
    txn = env.begin()

    try:
        seq = txn.get(entry_id.encode())
        if seq is None:
            return None
        return seq.decode()
    finally:
        env.close()


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


def extract_domain_structure(gt_structure_path, domain_info, output_pdb_path):
    try:
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
        return True
    except Exception as e:
        print(f"extract domain structure failed: {gt_structure_path}; {e}")
        return False


def process_single_domain(args_tuple):
    domain_seq, uniprot_ids, seq_lmdb_path, af_structure_dir, structures_dir = args_tuple
    domain_hash = compute_domain_hash(domain_seq)
    domain_pdb_path = Path(structures_dir) / f"{domain_hash}.pdb"

    if domain_pdb_path.exists():
        return {
            "status": "skip",
            "domain_hash": domain_hash,
            "domain_seq": domain_seq,
            "domain_pdb_path": str(domain_pdb_path),
        }

    failure_reasons = defaultdict(int)
    for uniprot_id in uniprot_ids:
        gt_structure_path = Path(af_structure_dir) / f"AF-{uniprot_id}-F1-model_v4.cif"
        if not gt_structure_path.exists():
            failure_reasons["gt_structure_not_found"] += 1
            continue

        gt_seq = read_gt_sequence(seq_lmdb_path, uniprot_id)
        if gt_seq is None:
            failure_reasons["gt_sequence_not_found"] += 1
            continue

        domain_info = domain_seq_to_domain_info(domain_seq, gt_seq)
        if domain_info is None:
            failure_reasons["domain_position_not_found"] += 1
            continue

        if extract_domain_structure(str(gt_structure_path), domain_info, str(domain_pdb_path)):
            return {
                "status": "success",
                "domain_hash": domain_hash,
                "domain_seq": domain_seq,
                "domain_pdb_path": str(domain_pdb_path),
            }
        failure_reasons["extract_failed"] += 1

    return {
        "status": "fail",
        "domain_hash": domain_hash,
        "domain_seq": domain_seq,
        "uniprot_ids": uniprot_ids,
        "reason": dict(failure_reasons),
    }


def decompress_gz_file(gz_path, output_path):
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return True
    except Exception:
        return False


def download_single_structure(args_tuple):
    uniprot_id, remote_host, remote_base_path, local_dir, max_retries = args_tuple
    gz_filename = f"AF-{uniprot_id}-F1-model_v4.cif.gz"
    cif_filename = f"AF-{uniprot_id}-F1-model_v4.cif"
    remote_path = os.path.join(remote_base_path, gz_filename)
    local_gz_path = os.path.join(local_dir, gz_filename)
    local_cif_path = os.path.join(local_dir, cif_filename)

    if os.path.exists(local_cif_path) and os.path.getsize(local_cif_path) > 0:
        return True, uniprot_id, "already_exists"

    rsync_cmd = [
        "rsync",
        "-avz",
        "--partial",
        "-e",
        "ssh -o StrictHostKeyChecking=no",
        f"{remote_host}:{remote_path}",
        local_gz_path,
    ]

    for attempt in range(max_retries):
        try:
            result = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=300)
        except Exception as e:
            if attempt == max_retries - 1:
                return False, uniprot_id, str(e)
            continue
        if result.returncode == 0 and os.path.exists(local_gz_path) and os.path.getsize(local_gz_path) > 0:
            if decompress_gz_file(local_gz_path, local_cif_path):
                os.remove(local_gz_path)
                return True, uniprot_id, "downloaded"
            return False, uniprot_id, "decompress_failed"
        if attempt == max_retries - 1:
            return False, uniprot_id, result.stderr.strip()

    return False, uniprot_id, "unknown_error"


def download_missing_structures(uniprot_ids, af_structure_dir, remote_host, remote_base_path, num_workers, max_retries):
    Path(af_structure_dir).mkdir(parents=True, exist_ok=True)
    tasks = [(uid, remote_host, remote_base_path, af_structure_dir, max_retries) for uid in sorted(uniprot_ids)]
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_uid = {executor.submit(download_single_structure, task): task[0] for task in tasks}
        for future in tqdm(as_completed(future_to_uid), total=len(future_to_uid), desc="Downloading AF2 structures"):
            results.append(future.result())
    return results


def prepare_domain_structures(args, domain_to_uniprots):
    output_dir = Path(args.domain_structures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    structures_dir = output_dir / "structures"
    structures_dir.mkdir(exist_ok=True)

    mapping_file = output_dir / "domain_hash_to_structure.json"
    seq_mapping_file = output_dir / "domain_hash_to_sequence.json"
    domain_hash_to_path = load_json(mapping_file)
    domain_hash_to_seq = load_json(seq_mapping_file)

    need_domains = []
    missing_uniprots = set()
    for domain_seq, uniprot_ids in domain_to_uniprots.items():
        domain_hash = compute_domain_hash(domain_seq)
        if domain_hash in domain_hash_to_path and Path(domain_hash_to_path[domain_hash]).exists():
            continue
        need_domains.append((domain_seq, uniprot_ids))
        for uniprot_id in uniprot_ids:
            structure_path = Path(args.af_structure_dir) / f"AF-{uniprot_id}-F1-model_v4.cif"
            if not structure_path.exists():
                missing_uniprots.add(uniprot_id)

    missing_uid_file = output_dir / "missing_uniprot_ids.txt"
    if missing_uniprots:
        with open(missing_uid_file, 'w') as f:
            for uniprot_id in sorted(missing_uniprots):
                f.write(f"{uniprot_id}\n")
        print(f"missing AF2 structures: {len(missing_uniprots)}; written to {missing_uid_file}")

        if args.download_missing:
            download_results = download_missing_structures(
                missing_uniprots,
                args.af_structure_dir,
                args.remote_host,
                args.remote_base_path,
                args.num_download_workers,
                args.max_retries,
            )
            failed = [item for item in download_results if not item[0]]
            print(f"download finished: {len(download_results) - len(failed)} success, {len(failed)} failed")

    if need_domains:
        process_args = [
            (domain_seq, uniprot_ids, args.seq_lmdb_path, args.af_structure_dir, str(structures_dir))
            for domain_seq, uniprot_ids in need_domains
        ]
        with Pool(processes=args.num_workers) as pool:
            results = list(tqdm(pool.imap(process_single_domain, process_args), total=len(process_args), desc="Extracting domains"))

        success_count = 0
        skip_count = 0
        fail_count = 0
        fail_reason_counts = defaultdict(int)
        fail_results = []
        for result in results:
            if result["status"] in ["success", "skip"]:
                domain_hash_to_path[result["domain_hash"]] = result["domain_pdb_path"]
                domain_hash_to_seq[result["domain_hash"]] = result["domain_seq"]
                if result["status"] == "success":
                    success_count += 1
                else:
                    skip_count += 1
            else:
                fail_count += 1
                fail_results.append(result)
                for reason, count in result.get("reason", {}).items():
                    fail_reason_counts[reason] += count
        print(f"domain extraction finished: {success_count} success, {skip_count} skip, {fail_count} failed")
        if fail_reason_counts:
            print(f"domain extraction failure reasons: {dict(fail_reason_counts)}")
        failure_file = output_dir / "domain_extraction_failures.json"
        dump_json(failure_file, {
            "reason_counts": dict(fail_reason_counts),
            "failures": fail_results,
        })
        print(f"domain extraction failure detail saved to {failure_file}")

    dump_json(mapping_file, domain_hash_to_path)
    dump_json(seq_mapping_file, domain_hash_to_seq)
    return domain_hash_to_path


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
        return None
    return max(scores)


def is_valid_tmscore(tm_score):
    if tm_score is None:
        return False
    try:
        return math.isfinite(float(tm_score))
    except (TypeError, ValueError):
        return False


def calculate_tmscore_task(args_tuple):
    key, pdb1, pdb2, tmalign_exec, tmp_dir, timeout = args_tuple
    fd, outpath = tempfile.mkstemp(prefix="tmp_tmalign_", suffix=".txt", dir=tmp_dir)
    os.close(fd)

    try:
        with open(outpath, 'w') as f:
            subprocess.run([tmalign_exec, pdb1, pdb2], stdout=f, stderr=subprocess.DEVNULL, timeout=timeout)
        tm_score = parse_tmalign_output(outpath)
        return key, tm_score
    except Exception as e:
        print(f"calculate TMscore failed: {pdb1} vs {pdb2}; {e}")
        return key, None
    finally:
        if os.path.exists(outpath):
            os.remove(outpath)


def make_pair_key(query_hash, candidate_hash):
    first, second = sorted([query_hash, candidate_hash])
    return f"{first}\t{second}"


def calculate_needed_tmscores(args, rows, domain_hash_to_path):
    if args.cache_path is None:
        cache_path = str(Path(args.output_path).with_suffix(".tmscore_cache.json"))
    else:
        cache_path = args.cache_path
    tmscore_cache = load_json(cache_path)

    task_by_key = {}
    cache_changed = False
    for row in rows:
        key = make_pair_key(row["query_hash"], row["candidate_hash"])
        if key in tmscore_cache and is_valid_tmscore(tmscore_cache[key]):
            continue
        query_path = domain_hash_to_path.get(row["query_hash"])
        candidate_path = domain_hash_to_path.get(row["candidate_hash"])
        if query_path is None or candidate_path is None:
            if tmscore_cache.get(key) is not None:
                tmscore_cache[key] = None
                cache_changed = True
            continue
        if not Path(query_path).exists() or not Path(candidate_path).exists():
            if tmscore_cache.get(key) is not None:
                tmscore_cache[key] = None
                cache_changed = True
            continue
        task_by_key[key] = (key, query_path, candidate_path, args.tmalign_exec, args.tmp_dir, args.timeout)

    if task_by_key:
        Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)
        tasks = list(task_by_key.values())
        with Pool(processes=args.num_workers) as pool:
            for key, tm_score in tqdm(pool.imap(calculate_tmscore_task, tasks), total=len(tasks), desc="Calculating TMscore"):
                tmscore_cache[key] = tm_score
                cache_changed = True

    if cache_changed:
        dump_json(cache_path, tmscore_cache)
        print(f"TMscore cache saved to {cache_path}")

    return tmscore_cache


def write_scored_tsv(output_path, header_line, header, rows, tmscore_cache):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if "Max_TMscore" in header:
        tmscore_idx = header.index("Max_TMscore")
        output_header = header_line
    else:
        tmscore_idx = len(header)
        output_header = header_line.rstrip("\n") + "\tMax_TMscore\n"

    with open(output_path, 'w') as f:
        f.write(output_header)
        for row in rows:
            line_list = row["line"].rstrip("\n").split("\t")
            key = make_pair_key(row["query_hash"], row["candidate_hash"])
            tm_score = tmscore_cache.get(key)
            if is_valid_tmscore(tm_score):
                tm_score_str = f"{float(tm_score):.6f}"
            else:
                tm_score_str = "nan"

            if tmscore_idx < len(line_list):
                line_list[tmscore_idx] = tm_score_str
            else:
                line_list.append(tm_score_str)
            f.write("\t".join(line_list) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/storage/yuanfajieLab/yuanfajie/sujin/Datasets/TED/embedding/afdb_cluster_power0.75/retrieval_results.tsv")
    parser.add_argument("--prefetched_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--af_structure_dir", type=str, default=AF_STRUCTURE_DIR)
    parser.add_argument("--domain_structures_dir", type=str, default=DOMAIN_STRUCTURES_DIR)
    parser.add_argument("--seq_lmdb_path", type=str, default=SEQ_LMDB_PATH)
    parser.add_argument("--tmalign_exec", type=str, default=TMALIGN_EXEC)
    parser.add_argument("--tmp_dir", type=str, default="/storage/yuanfajieLab/yuanfajie/tmpfile")
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--parse_cache_path", type=str, default=None)
    parser.add_argument("--force_reparse", action="store_true")
    parser.add_argument("--tmscore_threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--download_missing", action="store_true")
    parser.add_argument("--remote_host", type=str, default="TempCluster_yungu")
    parser.add_argument("--remote_base_path", type=str, default="/ssd/share-data/gzfile")
    parser.add_argument("--num_download_workers", type=int, default=8)
    parser.add_argument("--max_retries", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(0)

    header_line, header, rows, domain_to_uniprots = load_or_parse_retrieval_rows(args)
    print(f"matched retrieval rows: {len(rows)}")
    print(f"unique domains: {len(domain_to_uniprots)}")

    domain_hash_to_path = prepare_domain_structures(args, domain_to_uniprots)
    tmscore_cache = calculate_needed_tmscores(args, rows, domain_hash_to_path)
    write_scored_tsv(args.output_path, header_line, header, rows, tmscore_cache)

    valid_scores = []
    for row in rows:
        key = make_pair_key(row["query_hash"], row["candidate_hash"])
        tm_score = tmscore_cache.get(key)
        if is_valid_tmscore(tm_score):
            valid_scores.append(float(tm_score))

    qualified_num = sum(score < args.tmscore_threshold for score in valid_scores)
    print(f"valid TMscore rows: {len(valid_scores)}")
    print(f"TMscore < {args.tmscore_threshold}: {qualified_num}")
    print(f"writing into {args.output_path}")


if __name__ == "__main__":
    main()
