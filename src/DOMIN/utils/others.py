import torch
import numpy as np
import random
import time
import re
import os
import sys
import hashlib

from tqdm import tqdm
from Bio import SeqIO


# compute running time by 'with' grammar
class TimeCounter:
    def __init__(self, text: str = ""):
        self.text = text

    def __enter__(self):
        self.start = time.time()
        print(self.text, flush=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        t = end - self.start
        print(f"\nFinished. The running time is {t:.4f}s.\n", flush=True)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def progress_bar(now: int, total: int, desc: str = '', end=''):
    length = 50
    now = now if now <= total else total
    num = now * length // total
    progress_bar = '[' + '#' * num + '_' * (length - num) + ']'
    display = f'{desc:<10} {progress_bar} {int(now/total*100):02d}% {now}/{total}'

    print(f'\r\033[31m{display}\033[0m', end=end, flush=True)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


def random_seed():
    torch.seed()
    torch.cuda.seed()
    np.random.seed()
    random.seed()
    # torch.backends.cudnn.deterministic = False


def a3m_formalize(input, output, keep_gap=True):
    with open(output, 'w') as w:
        for record in SeqIO.parse(input, 'fasta'):
            desc = record.description
            if keep_gap:
                seq = re.sub(r"[a-z]", "", str(record.seq))
            else:
                seq = re.sub(r"[a-z-]", "", str(record.seq))
            w.write(f">{desc}\n{seq}\n")


def merge_file(file_list: list, save_path: str, drop_duplicates: bool = False, remove_subfiles: bool = False):
    """
    Merge multiple files into one file.
    Args:
        file_list: List of file paths to be merged.
        save_path: Path to save the merged file.
        drop_duplicates: If True, drop duplicate lines in the merged file.
        remove_subfiles: If True, remove the subfiles after merging them into the main file.
    """
    
    if drop_duplicates:
        tmp_set = set()

    with open(save_path, 'w') as w:
        for i, file in enumerate(file_list):
            with open(file, 'r') as r:
                for line in tqdm(r, f"Merging files... ({i+1}/{len(file_list)})"):
                    if drop_duplicates:
                        if line not in tmp_set:
                            w.write(line)
                            tmp_set.add(line)
                    else:
                        w.write(line)

            if remove_subfiles:
                os.remove(file)


def calculate_md5_file(file_path):
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()