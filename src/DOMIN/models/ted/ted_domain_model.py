import torch
import torch.distributed as dist
import torchmetrics
import math
import numpy as np
import faiss

from torchmetrics.functional import auroc
from ..saprot.base import SaprotBaseModel
from ..model_interface import register_model
from torch.nn.functional import normalize, cross_entropy


@register_model
class TedDomainModel(SaprotBaseModel):
    def __init__(self, temperature: float = 0.07, **kwargs):
        self.temperature = temperature
        self.domain2query = {}
        self.domain2key = {}
        self.domain2label = {}

        kwargs["load_pretrained"] = False
        super().__init__(task="base", **kwargs)
    
    def initialize_metrics(self, stage: str) -> dict:
        return_dict = {
            f"{stage}_acc": torchmetrics.Accuracy(),
        }
        
        return return_dict
    
    def initialize_model(self):
        super().initialize_model()
        
        # Add two MLP layers to encode proteins as query and key vectors
        hidden_size = self.model.config.hidden_size
        query_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size)
        )
        key_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size)
        )
        self.model.register_module("query_mlp", query_mlp)
        self.model.register_module("key_mlp", key_mlp)
        
        # Add learnable temperature
        temperature = torch.nn.Parameter(torch.tensor(self.temperature))
        self.model.register_parameter("temperature", temperature)

    def get_key_repr(self, seq: str or list) -> torch.Tensor:
        if isinstance(seq, str):
            seq = [seq]

        hidden_states = torch.stack(self.get_hidden_states_from_seqs(seq, reduction="mean"))
        key_states = self.model.key_mlp(hidden_states)
        key_states = normalize(key_states, dim=-1)
        return key_states

    def get_query_repr(self, seq: str or list) -> torch.Tensor:
        if isinstance(seq, str):
            seq = [seq]

        hidden_states = torch.stack(self.get_hidden_states_from_seqs(seq, reduction="mean"))
        query_states = self.model.query_mlp(hidden_states)
        query_states = normalize(query_states, dim=-1)
        return query_states

    def forward(self, inputs: dict, domains: list = None):
        """
        Args:
            inputs: A dictionary containing the input dict.
            domains: Domain names for the inputs. Used for evaluation.
        """
        # Compute hidden states. The first half of the hidden states are the first element of the pair,
        # and the second half are the second element of the pair.
        hidden_states = torch.stack(self.get_hidden_states_from_dict(inputs, reduction="mean"))
        
        query_states = self.model.query_mlp(hidden_states)
        key_states = self.model.key_mlp(hidden_states)
        
        # Normalize hidden states
        query_states = normalize(query_states, dim=-1)
        key_states = normalize(key_states, dim=-1)

        return query_states, key_states, domains
    
    def loss_func(self, stage: str, outputs, labels):
        query_states, key_states, domains = outputs
        device = query_states.device
        
        # Gather states from all GPUs
        all_query_states = self.all_gather(query_states).view(-1, query_states.shape[-1]).detach()
        all_key_states = self.all_gather(key_states).view(-1, key_states.shape[-1]).detach()
        
        # Compute similarity scores
        sim_scores = torch.matmul(query_states, all_key_states.T) / self.model.temperature
        
        # Create labels
        rank = dist.get_rank()
        half_batch = query_states.shape[0] // 2
        labels = torch.zeros(half_batch*2, dtype=torch.long, device=device)
        labels[:half_batch] = torch.arange(half_batch, device=device) + half_batch
        labels[half_batch:] = torch.arange(half_batch, device=device)
        labels += rank * half_batch * 2
        
        loss = cross_entropy(sim_scores, labels, ignore_index=-1)
        
        # Update metrics
        for metric in self.metrics[stage].values():
            metric.update(sim_scores, labels)
        
        if stage == "train":
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            # print(log_dict)

            self.log_info(log_dict)

            # Reset train metrics
            self.reset_metrics("train")
        
        else:
            # Create labels
            domain_labels = []
            for first_id, second_id in zip(domains[:half_batch], domains[half_batch:]):
                domain_labels.append([first_id, second_id])
                domain_labels.append([second_id, first_id])
            domain_labels = torch.tensor(domain_labels, dtype=torch.long, device=device)
            
            # Gather labels from all GPUs
            domain_labels = self.all_gather(domain_labels).view(-1, 2)
            for query_id, target_id in domain_labels:
                if query_id.item() not in self.domain2label:
                    self.domain2label[query_id.item()] = set()
                
                self.domain2label[query_id.item()].add(target_id.item())
            
            # Gather domain ids from all GPUs
            domain_ids = self.all_gather(domains).flatten()
            for domain_id, query_embed, key_embed in zip(domain_ids, all_query_states, all_key_states):
                if domain_id.item() not in self.domain2query:
                    self.domain2query[domain_id.item()] = query_embed
                    self.domain2key[domain_id.item()] = key_embed
            
        return loss
    
    def calculate_metrics(self):
        query_embeddings = torch.stack(list(self.domain2query.values()), dim=0)
        key_embeddings = torch.stack(list(self.domain2key.values()), dim=0)
        domain2rank = {idx: i for i, idx in enumerate(self.domain2query.keys())}
        rank2domain = {k: v for v, k in domain2rank.items()}
        
        sim_scores = torch.matmul(query_embeddings, key_embeddings.T)
        
        # Split domains evenly across GPUs
        domain_ids = list(self.domain2label.keys())
        n_gpus = dist.get_world_size()
        n_domain_per_gpu = math.ceil(len(domain_ids) / n_gpus)
        
        # Assign domains to GPUs to compute MAP
        curr_rank = dist.get_rank()
        batch_domain_ids = domain_ids[curr_rank * n_domain_per_gpu: (curr_rank + 1) * n_domain_per_gpu]
        
        # Calculate MAP and AUC
        ap_list = []
        auc_list = []
        for domain_id in batch_domain_ids:
            labels = self.domain2label[domain_id]
            query_idx = domain2rank[domain_id]
            scores = sim_scores[query_idx]
            rank_inds_for_query = torch.argsort(scores, descending=True)

            target_indices = torch.tensor([domain2rank[label] for label in labels], dtype=torch.long,
                                          device=sim_scores.device)

            mask = torch.isin(rank_inds_for_query, target_indices)
            hit_ranks = torch.nonzero(mask, as_tuple=False).squeeze().tolist()
            if isinstance(hit_ranks, int):
                hit_ranks = [hit_ranks]


            ap = np.mean([(i + 1) / (rank + 1) for i, rank in enumerate(hit_ranks)])
            ap_list.append(ap)
            # print(ap, domain_id, hit_ranks, target_indices, rank_inds_for_query.shape, auc_score, torch_auc)
            # if ap != 1 and dist.get_rank() == 0:
            #     print(ap, domain_id, hit_ranks, target_indices, rank_inds_for_query.shape, auc_score, torch_auc)
            #     print(target_indices)
            #     print(hit_ranks)
            #     print(rank_inds_for_query[:10])
            #     print(scores[rank_inds_for_query[:10]])
            #     for rk in rank_inds_for_query[:10]:
            #         print(rk, rank2domain[rk.item()])
            #     raise

            # Compute AUC
            labels = torch.zeros_like(scores, dtype=torch.long, device=sim_scores.device)
            labels[target_indices] = 1
            torch_auc = auroc(scores, labels)
            auc_list.append(torch_auc.item())

        
        batch_map = torch.tensor(ap_list, dtype=torch.float32, device=sim_scores.device).mean()
        batch_auc = torch.tensor(auc_list, dtype=torch.float32, device=sim_scores.device).mean()

        # Gather MAP and AUC from all GPUs
        map = self.all_gather(batch_map).mean().item()
        auc = self.all_gather(batch_auc).mean().item()
        
        # Reset domain2query, domain2key, and domain2label
        self.domain2query.clear()
        self.domain2key.clear()
        self.domain2label.clear()
        
        return map, auc
    
    def on_test_epoch_end(self):
        log_dict = self.get_log_dict("test")
        log_dict["test_loss"] = torch.cat(self.all_gather(self.test_outputs), dim=-1).mean()

        map, auc = self.calculate_metrics()
        log_dict["test_map"] = map
        log_dict["test_auc"] = auc

        if dist.get_rank() == 0:
            print(log_dict)
        self.log_info(log_dict)

        self.reset_metrics("test")

    def on_validation_epoch_end(self):
        log_dict = self.get_log_dict("valid")
        log_dict["valid_loss"] = torch.cat(self.all_gather(self.valid_outputs), dim=-1).mean()

        map, auc = self.calculate_metrics()
        log_dict["valid_map"] = map
        log_dict["valid_auc"] = auc

        if dist.get_rank() == 0:
            print(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_map"], mode="max")
        