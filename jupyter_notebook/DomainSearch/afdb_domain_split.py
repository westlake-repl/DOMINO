import sys

ROOT_DIR = f"{__file__.split('Pretraining')[0]}/Pretraining"
sys.path += [ROOT_DIR]

import gzip
import os
import argparse

from tqdm import tqdm
from data.parse import parse_structure
from Bio.PDB import PDBParser, FastMMCIFParser
from Bio.PDB.mmcifio import MMCIFIO
from utils.mpr import MultipleProcessRunnerSimplifier


mmcif_parser = FastMMCIFParser(QUIET=True)
mmcif_io = MMCIFIO()


def extract_and_concat_pdb_spans(
        input_path,
        output_path,
        chain_id,
        span: str) -> None:
    """
    Extract multiple residue spans from a protein structure,
    discard intermediate residues, and concatenate the kept parts.

    Args:
        input_path: Should be .cif.gz file.
        output_path: Should be .cif file.
        chain_id: Chain id to be processed.
        span: Residue spans to extract, e.g. "11-41_290-389".
              Residue indices start from 1 and refer to the order
              in the chain.
    """

    # -------- Parse span string --------
    # Example: "11-41_290-389" -> [(11, 41), (290, 389)]
    span_ranges = []
    for block in span.split("_"):
        start, end = block.split("-")
        span_ranges.append((int(start), int(end)))

    # -------- Read structure --------
    _, file = os.path.split(input_path)
    name, format = os.path.splitext(file)
    
    with gzip.open(input_path, 'rt') as cif:
        structure = mmcif_parser.get_structure(name, cif)

    # -------- Process chain --------
    for chain in structure[0]:
        if chain.id != chain_id:
            continue

        residues_to_remove = []
        cnt = 1  # residue index starts from 1

        for residue in chain:
            keep = False
            for start, end in span_ranges:
                if start <= cnt <= end:
                    keep = True
                    break

            if not keep:
                residues_to_remove.append(residue.id)

            cnt += 1

        # -------- Remove unwanted residues --------
        for residue_id in residues_to_remove:
            chain.detach_child(residue_id)

        # -------- Write output --------
        mmcif_io.set_structure(chain)
        mmcif_io.save(output_path)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()

    interval = args.end - args.start

    gz_dir = "/ssd/share-data/gzfile"
    domain_split_file = "/share/home/yuanfajieLab/zhouxibin/sujin/Datasets/TED/data_domain_ge2.tsv"

    # Split structures into
    save_dir = f"/ssd/yuanfajieLab/zhouxibin/TED/split_domains"
    sub_dir = 0
    cnt = 0
    
    with open(domain_split_file, "r") as r:
        domain_splits = []
        for i, line in tqdm(enumerate(r)):
            os.makedirs(f"{save_dir}/{sub_dir}", exist_ok=True)
            uniprot_id, spans = line.strip().split("\t")
            
            for span in spans.strip().split(":"):
                pdb_path = f"{gz_dir}/AF-{uniprot_id}-F1-model_v4.cif.gz"
                save_path = f"{save_dir}/{sub_dir}/{uniprot_id}_{span}.cif"

                cnt += 1
                if cnt == 50000:
                    sub_dir += 1
                    cnt = 0
            
                if args.start <= i < args.end:
                    domain_splits.append([pdb_path, save_path, span])
                
            if i >= args.end:
                break
            
    def do(process_id, idx, item, writer):
        pdb_path, save_path, span = item
        if os.path.exists(save_path):
            seq = parse_structure(save_path, "A")["A"]["seq"]
            writer.write(f"{seq}\t{save_path}\n")
            return

        try:
            extract_and_concat_pdb_spans(pdb_path, save_path, "A", span)
            seq = parse_structure(save_path, "A")["A"]["seq"]
            writer.write(f"{seq}\t{save_path}\n")

        except Exception as e:
            print(f"Error processing {pdb_path} with span {span}: {e}")

    mapping_path = f"{save_dir}/mapping_{args.start}_{args.end}.tsv"
    mprs = MultipleProcessRunnerSimplifier(domain_splits, do, n_process=1, save_path=mapping_path)
    mprs.run()


if __name__ == '__main__':
    # main()
    save_dir = f"/ssd/yuanfajieLab/zhouxibin/TED/split_domains"
    cnt = 0
    for sub_dir in tqdm(os.listdir(save_dir)):
        # if sub_dir.endswith(".tsv"):
        #     print(sub_dir)
        #     cnt += 1

        if not sub_dir.startswith("map"):
            num_files = len(os.listdir(f"{save_dir}/{sub_dir}"))
            cnt += num_files
            # break

    print(cnt)