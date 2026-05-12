# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DOMINO** (Learning Domain Co-occurrence for Multidomain Protein Design) is a protein engineering framework for designing multi-domain proteins. This is important for drug discovery and synthetic biology where proteins with multiple functional domains are needed.

The project consists of two complementary models:

### DOMIN（Domain Interaction Model）
- **Task**: Evaluates whether two protein domains can co-occur in the same protein (domain compatibility scoring)
- **Method**: Contrastive learning with Query-Key representation learning
- **Scoring**: `similarity = dot_product(query_repr, key_repr) / temperature`
- **Key innovation**: Learnable temperature parameter for calibrated similarity scores
- **Use case**: Predict domain compatibility before generating full sequences

### DOMO（Domain Generation Model）
- **Task**: Given a list of domain sequences, generate a complete multi-domain protein sequence
- **Method**: ESM encoder + Qwen3-0.6B with cross-attention
- **Architecture**:
  1. Each domain is independently encoded by ESM encoder
  2. Domain embeddings are concatenated along sequence dimension
  3. Qwen3-0.6B generates the full sequence using cross-attention over domain features
- **Note**: Limited to ~1024 tokens of concatenated domain context

## Environment Setup

```bash
conda env create -f requirement.yml
conda activate DOMINO_env
```

Model weights download (from HuggingFace `westlake-repl/DOMINO`):
```bash
pip install -U huggingface_hub
hf download westlake-repl/DOMINO --repo-type model --local-dir weights/
```

## Inference Commands

```bash
# DOMIN inference (structural matching score)
python DOMIN_inference.py

# DOMO inference (domain combination generation)
python DOMO_inference.py
```

## Architecture

### DOMIN (`src/DOMIN/`)
- Built on ESM/Saprot backbone via `SaprotBaseModel` (extends `AbstractModel` → PyTorch Lightning)
- `TedDomainModel`: Adds query_mlp/key_mlp projection heads and learnable temperature
- `get_query_repr()` / `get_key_repr()` return L2-normalized embeddings
- Similarity = `dot_product(query, key) / temperature`

### DOMO (`src/DOMO/`)
- `BaseModel`: Simple torch.nn.Module base with logger
- `Qwen3CAwDomainConditioning`: Qwen3-0.6B with ESM encoder as cross-attention context
- `forward_encoder()`: ESM encodes domain sequences, projects to Qwen3 hidden size
- Generation uses cross-attention over ESM domain features

### Common Patterns
- `src/DOMIN/models/saprot/base.py`: ESM backbone loader with LoRA support
- `src/DOMO/utils/init_utils.py`: `construct_class_by_name()` for dynamic class instantiation
- Both use OmegaConf for config management

## Config Files
- DOMIN: `src/DOMIN/configs/DOMIN_config.yaml`
- DOMO: `src/DOMO/configs/DOMO_config.yaml`

## Weights Structure
```
weights/
├── DOMIN/DOMIN.pt
└── DOMO/pytorch_model.bin
```

## Working Directory

Use `claude_workspace/` for any tasks given to Claude Code.
