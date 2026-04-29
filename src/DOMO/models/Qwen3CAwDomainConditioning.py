import re
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import EsmTokenizer, EsmModel, GenerationConfig, Qwen3Config
from typing import Optional

import sys
sys.path.append("./src/DOMO/models")
from BaseModel import BaseModel
from Qwen3.modeling_domainconditioning_qwen3 import Qwen3CAForCausalLM
from Qwen3.configuration_domainconditioning_qwen3 import Qwen3Config
from utils.init_utils import construct_class_by_name

default_generation_config = GenerationConfig(
    max_length=1024,
    do_sample=True,
    temperature=0.8,
)


class Qwen3CAwDomainConditioning(BaseModel):
    def __init__(
        self,
        qwen3_type="Qwen/Qwen3-100M",
        esm_type="facebook/esm2_t12_35M_UR50D",
        criterion_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.logger.info(
            f"{self.__class__.__name__} initialized with gpt_type: {qwen3_type}, esm_type: {esm_type}"
        )

        self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
        # Since we need to use the cross attention, we need to set the use_cache to False and is_decoder to True
        if qwen3_type == "Qwen/Qwen3-100M":
            # 创建100M配置
            config = Qwen3Config(
                architectures=["Qwen3ForCausalLM"],
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=self.tokenizer.cls_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                head_dim=64,
                hidden_act="silu",
                hidden_size=768,
                initializer_range=0.02,
                intermediate_size=2304,
                max_position_embeddings=40960,
                max_window_layers=16,
                model_type="qwen3",
                num_attention_heads=12,
                num_hidden_layers=16,
                num_key_value_heads=6,
                rms_norm_eps=1e-06,
                rope_scaling=None,
                rope_theta=1000000,
                sliding_window=None,
                tie_word_embeddings=True,
                torch_dtype="bfloat16",
                use_cache=True,  #
                use_sliding_window=False,
                vocab_size=len(self.tokenizer.get_vocab()),
                add_cross_attention=True,
            )
            # 创建模型
            self.qwen3 = Qwen3CAForCausalLM(config)
        else:
            config = Qwen3Config.from_pretrained(qwen3_type, add_cross_attention=True, vocab_size=len(self.tokenizer.get_vocab()))
            self.qwen3 = Qwen3CAForCausalLM(config)


        self.qwen3.gradient_checkpointing_enable()

        self.esm_encoder = EsmModel.from_pretrained(esm_type)
        self.esm_encoder.encoder.gradient_checkpointing = True
        self.esm_encoder.gradient_checkpointing_enable()

        self.esm_encoder.pooler.dense.weight.requires_grad = False
        self.esm_encoder.pooler.dense.bias.requires_grad = False
        self.esm_encoder.contact_head.regression.weight.requires_grad = False
        self.esm_encoder.contact_head.regression.bias.requires_grad = False
        # self.esm_encoder.embeddings.position_embeddings.weight.requires_grad = False

        self.domain_feats_projector = nn.Linear(
            self.esm_encoder.config.hidden_size, self.qwen3.config.hidden_size
        )
        if criterion_kwargs is not None:
            self.criterion = construct_class_by_name(
                **criterion_kwargs, logger=self.logger
            )
        else:
            self.criterion = None

    def _forward(
        self,
        seq_ids: torch.Tensor,
        domain_ids: torch.Tensor,
        domain_masks: torch.Tensor,
        num_domains_per_protein: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> dict:
        domain_feats, domain_feat_masks = self.forward_encoder(
            domain_ids, domain_masks, num_domains_per_protein
        )

        out_dict = self.qwen3(
            input_ids=seq_ids,
            encoder_hidden_states=domain_feats,
            encoder_attention_mask=domain_feat_masks,
            labels=labels,
            return_dict=True,
        )
        return out_dict  # [bs, seq_len, vocab_size]

    def forward(
        self,
        seq_ids: torch.Tensor,
        domain_ids: torch.Tensor,
        domain_masks: torch.Tensor,
        num_domains_per_protein: torch.Tensor,
        **kwargs,
    ):
        """
        this function is used for training, computing the loss
        seq_ids: [bs, seq_len]
        kwargs: other arguments

        return:
            logits: [bs, seq_len, vocab_size]
            target: [bs, seq_len]
            loss_mask: [bs, seq_len]
            loss: the loss value
        """
        target = seq_ids.clone()
        target = target.masked_fill(target == self.tokenizer.pad_token_id, -100)

        out_dict = self._forward(
            seq_ids, domain_ids, domain_masks, num_domains_per_protein, labels=target
        )
        logits = out_dict["logits"]
        if self.criterion:
            domain_mask = self._create_domain_mask(seq_ids, kwargs["domain_positions"])
            loss, logging_output = self.criterion(logits, target, domain_mask)
        else:
            loss = out_dict["loss"]
            logging_output = {}
        return {"logits": logits, "target": target, "loss": loss, **logging_output}

    def _create_domain_mask(
        self,
        seq_ids: torch.Tensor,
        domain_positions: list[list[tuple[int, int]]],
    ) -> torch.Tensor:
        """
        根据domain信息创建domain_mask
        Args:
            seq_ids: [bs, seq_len] 序列
            domain_positions: bs, num_domain, domain_positions
        Returns:
            domain_mask: [bs, seq_len] 1表示domain区域，0表示非domain区域
        """
        batch_size, seq_len = seq_ids.shape
        domain_mask = torch.zeros(batch_size, seq_len, device=seq_ids.device)

        for i, positions in enumerate(domain_positions):
            for start, end in positions:
                domain_mask[i, start + 1 : end + 1] = (
                    1  # shift by 1 to exclude the bos token
                )

        return domain_mask

    def forward_encoder(
        self,
        domain_ids: torch.Tensor,
        domain_masks: torch.Tensor,
        num_domains_per_protein: torch.Tensor,
    ):

        encoder_out = self.esm_encoder(domain_ids, domain_masks).last_hidden_state
        encoder_emb_dim = encoder_out.shape[-1]

        # 第四步：按照原始结构重新组织数据
        domain_feats = []
        domain_feat_masks = []

        start_idx = 0
        for i, domain_count in enumerate(num_domains_per_protein):
            end_idx = start_idx + domain_count

            # 提取当前data point的encoder输出
            current_encoder_out = encoder_out[
                start_idx:end_idx
            ]  # [num_domain, seq_len, hidden_size]
            current_domain_masks = domain_masks[
                start_idx:end_idx
            ]  # [num_domain, seq_len]

            # 重塑并去除padding
            encoder_out_flat = current_encoder_out.reshape(
                -1, encoder_emb_dim
            )  # [num_domain*seq_len, hidden_size]
            domain_masks_flat = current_domain_masks.reshape(-1)  # [num_domain*seq_len]

            # 去除padding
            bool_mask = domain_masks_flat.bool()
            encoder_out_flat = encoder_out_flat[bool_mask]
            domain_masks_flat = domain_masks_flat[bool_mask]

            domain_feats.append(encoder_out_flat)
            domain_feat_masks.append(domain_masks_flat)

            start_idx = end_idx

        # 第五步：重新padding到1024长度
        domain_feats_padded = []
        domain_feat_masks_padded = []

        for domain_feat, domain_feat_mask in zip(domain_feats, domain_feat_masks):
            if domain_feat.shape[0] < 1024:
                domain_feat = F.pad(
                    domain_feat,
                    (0, 0, 0, 1024 - domain_feat.shape[0]),
                    mode="constant",
                    value=0,
                )
                domain_feat_mask = F.pad(
                    domain_feat_mask,
                    (0, 1024 - domain_feat_mask.shape[0]),
                    mode="constant",
                    value=0,
                )
            else:
                domain_feat = domain_feat[:1024, :]  # [1024, hidden_size]
                domain_feat_mask = domain_feat_mask[:1024]  # [1024]

            domain_feats_padded.append(domain_feat)
            domain_feat_masks_padded.append(domain_feat_mask)

        # 第六步：stack并投影
        domain_feats = torch.stack(domain_feats_padded)  # [bs, 1024, hidden_size]
        domain_feats = self.domain_feats_projector(
            domain_feats
        )  # [bs, 1024, qwen_hidden_size]
        domain_feat_masks = torch.stack(domain_feat_masks_padded)  # [bs, 1024]

        return domain_feats, domain_feat_masks

    def initialize_output_tokens(self, bs: int, **kwargs):

        start_id = self.tokenizer.cls_token_id
        input_ids = (
            (torch.zeros((1)) + start_id).unsqueeze(0).repeat(bs, 1)
        )  # create batch dim
        input_ids = input_ids.to(torch.long)
        input_ids = input_ids.to(next(self.parameters()).device)
        return input_ids

    def to_list(self, seq: torch.Tensor):
        return [
            seq[i, ...].detach().cpu().numpy().tolist() for i in range(seq.shape[0])
        ]

    def clean_and_format_seq(self, seq: list[str]):
        cleaned_data = []
        for item in seq:
            processed_string = re.sub(r"<cls>", "", item)
            processed_string = re.sub(r"<eos>", "", processed_string)
            processed_string = re.sub(r"<pad>", "", processed_string)
            processed_string = processed_string.replace(" ", "")
            cleaned_data.append(processed_string)
        return cleaned_data

    def generate(
        self,
        domain_ids: torch.Tensor,
        domain_masks: torch.Tensor,
        num_domains_per_protein: torch.Tensor,
        generation_config: GenerationConfig = default_generation_config,
        verbose=True,
        **kwargs,
    ):
        # override the prepare_inputs_for_generation to use the cross attention
        # self.qwen3.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        # 0) encoding
        encoder_out, encoder_mask = self.forward_encoder(
            domain_ids, domain_masks, num_domains_per_protein
        )

        generation_config.bos_token_id = self.tokenizer.cls_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        # 1) initialized from all mask tokens
        # initial_output_tokens = self.initialize_output_tokens(bs=encoder_out.shape[0])
        sample_results = self.qwen3.generate(
            generation_config=generation_config,
            num_return_sequences=1,
            encoder_hidden_states=encoder_out,
            encoder_attention_mask=encoder_mask,
            return_dict_in_generate=True,
        )
        tokens = self.to_list(sample_results.sequences)
        sequences = self.tokenizer.batch_decode(tokens)
        return {
            "output_seqs": self.clean_and_format_seq(sequences),
        }
