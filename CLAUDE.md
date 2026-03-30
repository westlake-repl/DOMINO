# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DomainReComb is a two-step protein domain recombination system:
- **Step 1: Domain Searching** - Uses contrastive learning models (TED) to search and identify protein domains
- **Step 2: Domain Combination** - Uses transformer-based models (Qwen3 + ESM2) to generate novel protein sequences by combining domains

## Environment Setup

Install dependencies:
```bash
pip install -r reqs_domaincomb.txt
```

Key dependencies: PyTorch 2.6.0, transformers 4.51.0, ESM 2.0.0, flash-attn 2.8.3, accelerate, deepspeed, ray

## Running Domain Combination

Basic usage:
```bash
python domaincomb.py --config src/DomainComb/configs/01-AR-esm2-qwen3-200M.yaml
```

Available configs:
- `01-AR-esm2-qwen3-200M.yaml` - 200M parameter model
- `02-AR-esm2-qwen3-1.2B.yaml` - 1.2B parameter model

## Architecture

### DomainComb Module (src/DomainComb/)

**Core Model Architecture:**
- `models/Qwen3CAwDomainConditioning.py` - Main model combining ESM2 encoder with Qwen3 decoder using cross-attention
  - ESM2 encoder processes input domains and generates domain embeddings
  - Domain features are projected and fed to Qwen3 via cross-attention
  - Qwen3 generates the combined protein sequence autoregressively
  - Uses gradient checkpointing for memory efficiency

**Model Components:**
- `models/BaseModel.py` - Base class for all DomainComb models
- `models/Qwen3/modeling_domainconditioning_qwen3.py` - Custom Qwen3 with cross-attention support
- `models/Qwen3/configuration_domainconditioning_qwen3.py` - Qwen3 configuration

**Utilities:**
- `utils/init_utils.py` - Dynamic class construction via `construct_class_by_name()` - used throughout for instantiating models from config strings

### DomainSearch Module (src/DomainSearch/)

**Core Architecture:**
- `models/abstract_model.py` - PyTorch Lightning base class for all DomainSearch models
  - Handles training loop, optimizer/scheduler setup, checkpointing
  - Implements metric tracking across train/valid/test stages
  - Uses custom learning rate schedulers from `utils/lr_scheduler.py`

**Models:**
- `models/ted/ted_domain_model.py` - TED (Two-Encoder Domain) model for contrastive learning
  - Dual MLP heads (query/key) for contrastive representation learning
  - Uses normalized embeddings with learnable temperature
  - Supports domain retrieval via FAISS indexing
- `models/saprot/` - SaProt-based models with PEFT support
  - `base.py` - Base SaProt model wrapper
  - `self_peft/` - Custom PEFT (Parameter-Efficient Fine-Tuning) implementation

**Model Registration:**
- `models/model_interface.py` - Dynamic model loading via `@register_model` decorator and `ModelInterface.init_model()`

**Utilities:**
- `utils/generate_lmdb.py` - LMDB database generation for efficient data loading
- `utils/lr_scheduler.py` - Custom learning rate schedulers
- `utils/mpr.py` - Multi-process related utilities
- `utils/others.py` - Miscellaneous utilities including TimeCounter

### Large Scale Design (large_scale_design/scirpts/)

**Distributed Inference:**
- `run_ray.py` - Ray-based distributed inference for large-scale protein generation
  - Uses Ray workers with GPU allocation
  - Implements length-based batching for efficiency
  - Supports seed setting for reproducibility

**Evaluation:**
- `evaluate/cal_domain_recovery.py` - Calculate domain recovery metrics
- `evaluate/random_sample_n_res.py` - Random sampling utilities
- `evaluate/mmseqs_ur100.sh` - MMseqs2 clustering against UR100

**Slurm Scripts:**
- `Slurm_launch_public_h800.sh` - Single node H800 GPU job
- `Slurm_launch_public_h800_2nodes.sh` - Multi-node H800 GPU job
- `run.sh`, `run_2nodes.sh` - General execution scripts

**Data Processing:**
- `dataloader.py` - Step2Dataset for domain combination data loading
- `convert2fasta.py` - Convert sequences to FASTA format

## Key Patterns

**Dynamic Class Construction:**
All models use `construct_class_by_name()` from `init_utils.py`. Config files specify models as:
```yaml
Model:
  kwargs:
    class_name: models.Qwen3CAwDomainConditioning.Qwen3CAwDomainConditioning
```

**Model Loading:**
- DomainComb models: Load via `torch.load()` and `model.load_state_dict()`
- DomainSearch models: Use PyTorch Lightning checkpointing via `AbstractModel.load_checkpoint()`

**Tokenization:**
All models use ESM2 tokenizer: `EsmTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")`

## Checkpoints

Checkpoints should be placed in `checkpoints/` directory (not included in repo):
```
checkpoints/
  DomainComb/
    01-AR-esm2-qwen3-200M/pytorch_model.bin
    02-AR-esm2-qwen3-1.2B/pytorch_model.bin
```

Note: The README mentions copying checkpoints via `cp -r /yuanfajie/DomainReComb/checkpoints .`

## Code Style Notes

- Uses PyTorch Lightning for DomainSearch training infrastructure
- Gradient checkpointing enabled for memory efficiency
- Flash attention support (flash-attn 2.8.3)
- Mixed precision training support via accelerate/deepspeed
- Path handling: Uses `sys.path.append()` for module imports (see `domaincomb.py`, `run_ray.py`)
