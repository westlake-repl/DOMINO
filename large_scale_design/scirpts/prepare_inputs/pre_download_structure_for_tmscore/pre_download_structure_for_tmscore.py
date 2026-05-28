import argparse
import gzip
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


INPUT_PATH = "/storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb/large_scale_design/inputs/LargeScale-02-0324-retrieval_results_random5000_backup.tsv"
LOCAL_DIR = "/storage/yuanfajieLab/yuanfajie/fengyuan/DomainReComb/.cache/af_structures"
REMOTE_HOST = "TempCluster_yungu"
REMOTE_DIR = "/ssd/share-data/gzfile"


def get_column_idx(header, name, default_idx):
    if name in header:
        return header.index(name)
    return default_idx


def collect_uids(input_path, top_k):
    gt_uids = set()
    candidate_uids = set()
    query_counts = {}

    with open(input_path, 'r') as f:
        header = f.readline().rstrip("\n").split("\t")
        query_idx = get_column_idx(header, "Query_Segment", 0)
        gt_idx = get_column_idx(header, "Ground_Truth_Uniprot_IDs", 5)
        candidate_idx = get_column_idx(header, "Candidate_Uniprot_IDs", len(header) - 1)

        for line in tqdm(f, desc="Collecting uids"):
            line_list = line.rstrip("\n").split("\t")
            if len(line_list) <= max(query_idx, gt_idx, candidate_idx):
                raise ValueError(f"malformed input line: {line[:200]}")

            query_seg = line_list[query_idx]
            query_counts[query_seg] = query_counts.get(query_seg, 0) + 1

            gt_uids.update(uid.strip() for uid in line_list[gt_idx].split(",") if uid.strip())
            if query_counts[query_seg] <= top_k:
                candidate_uids.update(uid.strip() for uid in line_list[candidate_idx].split(",") if uid.strip())

    return gt_uids, candidate_uids


def decompress_gz_file(gz_path, output_path):
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def download_one(uid, args):
    local_dir = Path(args.local_dir)
    cif_filename = f"AF-{uid}-F1-model_v4.cif"
    gz_filename = f"{cif_filename}.gz"
    local_cif_path = local_dir / cif_filename
    local_gz_path = local_dir / gz_filename

    if local_cif_path.exists() and local_cif_path.stat().st_size > 0:
        return uid, "exists", None

    remote_path = f"{args.remote_host}:{args.remote_dir}/{gz_filename}"
    result = subprocess.run(
        ["scp", "-q", remote_path, str(local_dir)],
        capture_output=True,
        text=True,
        timeout=args.timeout,
    )
    if result.returncode != 0:
        return uid, "download_failed", result.stderr.strip()

    if not local_gz_path.exists() or local_gz_path.stat().st_size == 0:
        return uid, "download_empty", str(local_gz_path)

    try:
        decompress_gz_file(local_gz_path, local_cif_path)
        local_gz_path.unlink()
    except Exception as e:
        return uid, "decompress_failed", str(e)

    if not local_cif_path.exists() or local_cif_path.stat().st_size == 0:
        return uid, "decompress_empty", str(local_cif_path)

    return uid, "downloaded", None


def write_uid_file(path, uids):
    with open(path, 'w') as f:
        for uid in sorted(uids):
            f.write(f"{uid}\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=INPUT_PATH)
    parser.add_argument("--local_dir", type=str, default=LOCAL_DIR)
    parser.add_argument("--remote_host", type=str, default=REMOTE_HOST)
    parser.add_argument("--remote_dir", type=str, default=REMOTE_DIR)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--uid_file", type=str, default=None)
    parser.add_argument("--failed_uid_file", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    gt_uids, candidate_uids = collect_uids(args.input_path, args.top_k)
    all_uids = gt_uids | candidate_uids
    existing_uids = {
        path.name.removeprefix("AF-").removesuffix("-F1-model_v4.cif")
        for path in local_dir.glob("AF-*-F1-model_v4.cif")
        if path.stat().st_size > 0
    }
    missing_uids = sorted(all_uids - existing_uids)

    print(f"GT uids: {len(gt_uids)}")
    print(f"top{args.top_k} candidate uids: {len(candidate_uids)}")
    print(f"total uids: {len(all_uids)}")
    print(f"already exists: {len(all_uids & existing_uids)}")
    print(f"to download: {len(missing_uids)}")

    uid_file = args.uid_file
    if uid_file is None:
        uid_file = str(local_dir / f"tmscore_prefetch_top{args.top_k}_uids.txt")
    write_uid_file(uid_file, missing_uids)
    print(f"uids to download written to {uid_file}")

    if args.dry_run:
        return

    status_counts = {}
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_one, uid, args) for uid in missing_uids]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading structures"):
            uid, status, detail = future.result()
            status_counts[status] = status_counts.get(status, 0) + 1
            if status not in {"exists", "downloaded"}:
                failed.append((uid, status, detail))

    print(f"status: {status_counts}")

    failed_uid_file = args.failed_uid_file
    if failed_uid_file is None:
        failed_uid_file = str(local_dir / f"tmscore_prefetch_top{args.top_k}_failed_uids.txt")
    with open(failed_uid_file, 'w') as f:
        for uid, status, detail in failed:
            f.write(f"{uid}\t{status}\t{detail or ''}\n")

    print(f"failed: {len(failed)}")
    print(f"failed details written to {failed_uid_file}")

    if failed:
        raise RuntimeError(f"{len(failed)} structure downloads failed")


if __name__ == "__main__":
    main()
