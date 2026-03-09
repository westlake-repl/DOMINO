from typing import Dict, Any
import torch
from torch.utils.data import DataLoader
from transformers import EsmTokenizer
import sys
sys.path.append("/storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain")
from utils.dataloader_utils import domain_fillin


def remove_lower_alpha(text: str) -> str:
    text_seg = text.split("<unk>")
    text_seg_without_lower = []
    for seg in text_seg:
        seg_list = list(seg)
        seg_list_without_lower = []
        for char in seg_list:
            if char.isupper():
                seg_list_without_lower.append(char)
        text_seg_without_lower.append("".join(seg_list_without_lower))
    return "<unk>".join(text_seg_without_lower)


class Step2Dataset():
    def __init__(self, tsv_path: str, tokenizer: EsmTokenizer):
        # super().__init__(**kwargs)
        self.tsv_path = tsv_path
        self.tokenizer = tokenizer
        domain_pair_list = []
        with open(tsv_path, "r") as f:
            first_line = f.readline()
            for line in f:
                query_domain_seq, retrieval_domain_seq = line.strip().split("\t")[:2]
                query_domain_seq = remove_lower_alpha(query_domain_seq)
                retrieval_domain_seq = remove_lower_alpha(retrieval_domain_seq)
                domain_pair_list.append((query_domain_seq, retrieval_domain_seq))
        # self.domain_pair_list = domain_pair_list
        self.index_mapper = domain_pair_list

    def __len__(self):
        return len(self.index_mapper)
        
    def __getitem__(self, idx: int) -> Dict[str, str | list[str]]:
        ## domain pieces
        domain_pieces = []
        for single_domain in self.index_mapper[idx]:
            domain_piece = single_domain.split("<unk>")
            domain_pieces.extend(domain_piece)
        return {"domain": self.index_mapper[idx], "domain_pieces": domain_pieces}

    def collate(self, batch: list[Dict[str, Any]]) -> Dict[str, Any]:
        """
        for domain, we return a list (which length is the batch size), each element is a tensor of shape [num_domain, seq_len]
        """
        keys = [key for key in batch[0].keys()]
        # dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        dict_batch = {}
        # Here we modify structure token by plddt mask
        for k in keys:
            dict_batch[k] = [dic[k] if k in dic else None for dic in batch]

        # encode seq and domain
        if self.tokenizer is not None:
            if "seq" in dict_batch:
                encodings = self.tokenizer(
                    dict_batch["seq"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_aa_seq_len,
                    padding="longest",  # max_length,longest
                )
                dict_batch.update(
                    {
                        "seq_ids": encodings.input_ids,
                        "seq_masks": encodings.attention_mask,
                    }
                )

            num_domains_per_protein = torch.tensor(
                [
                    len(domains_per_protein)
                    for domains_per_protein in dict_batch["domain"]
                ]
            )

            # calculate num_domain_pieces_per_protein
            if "domain_positions" in dict_batch:
                num_domain_pieces_per_protein = torch.tensor(
                    [
                        len(domain_position_list)
                        for domain_position_list in dict_batch["domain_positions"]
                    ]
                )
            else:
                num_domain_pieces_per_protein = []
                for domain_list in dict_batch["domain"]:
                    cur_domain_pieces_num = 0
                    for domain in domain_list:
                        cur_domain_pieces_num += len(domain.split("<unk>"))
                    num_domain_pieces_per_protein.append(cur_domain_pieces_num)
                num_domain_pieces_per_protein = torch.tensor(num_domain_pieces_per_protein)

            # encode domain
            all_domain_list = []
            for domains_per_protein in dict_batch["domain"]:
                all_domain_list.extend(domains_per_protein)

            encodings = self.tokenizer(
                all_domain_list,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="longest",  # max_length,longest
            )
            domain_ids = encodings.input_ids
            domain_masks = encodings.attention_mask

            dict_batch.update(
                {
                    "domain_ids": domain_ids,
                    "domain_masks": domain_masks,
                    "num_domains_per_protein": num_domains_per_protein,
                    "num_domain_pieces_per_protein": num_domain_pieces_per_protein,
                }
            )

            # encode domain pieces
            if "domain_pieces" in dict_batch and dict_batch["domain_pieces"][0] is not None:
                all_domain_pieces = []
                for domain_pieces in dict_batch["domain_pieces"]:
                    all_domain_pieces.extend(domain_pieces)
                encodings = self.tokenizer(
                    all_domain_pieces,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding="longest",  # max_length,longest
                )
                domain_pieces_ids = encodings.input_ids
                domain_pieces_masks = encodings.attention_mask
                dict_batch.update(
                    {
                        "domain_pieces_ids": domain_pieces_ids,
                        "domain_pieces_masks": domain_pieces_masks,
                    }
                )

        return dict_batch


class GenerationDataModule():
    def __init__(self, **kwargs):
        self.tsv_path = kwargs.pop("tsv_path")
        self.tokenizer = kwargs.pop("tokenizer")
        self.eval_batch_size = kwargs.pop("eval_batch_size")
        self.num_workers_per_gpu = kwargs.pop("num_workers_per_gpu")

    def set_test_dataset(self):
        self.test_dataset = Step2Dataset(
            tsv_path=self.tsv_path, tokenizer=self.tokenizer
        )

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers_per_gpu,
            pin_memory=True,
            collate_fn=self.test_dataset.collate,
        )
        return loader

