## Environment setup

### Method 1: Using `requirement.yml` (Recommended)
We provide a `requirement.yml` file to help you easily set up the environment. Please run the following script to create and activate the conda environment:
```bash
conda env create -f requirement.yml
conda activate DOMINO_env
```

### Method 2: Using `requirement.txt`
If the first method fails or you prefer using pip, you can manually create a Conda environment and install the dependencies using the provided `requirement.txt` file:
```bash
conda create -n DOMINO_env python=3.11 -y
conda activate DOMINO_env
pip install -r requirement.txt
```


## Download model weights

We have released the DOMINO model weights on Hugging Face: **[DOMINO](https://huggingface.co/westlake-repl/DOMINO)**

To download the DOMINO model weights into the `weights` directory, please install the `huggingface_hub` package and run the following script:
```bash
pip install -U huggingface_hub
hf download westlake-repl/DOMINO --repo-type model --local-dir weights/
```

After downloading, your directory structure should look like this:
```bash
DOMINO/
├── weights/
│   ├── DOMIN/
│   ├── DOMO/
│   └── ...
├── src/
├── DOMIN_inference.py
├── DOMO_inference.py
└── ...
```


## Inference with DOMIN
You can quickly run the DOMIN inference code using the provided script:
```bash
python DOMIN_inference.py
```

DOMIN is designed to evaluate the structural compatibility and matching degree of protein sequences. The core inference process involves extracting the Query and Key representations from a given structural sequence and computing their temperature-scaled dot product to derive the final matching score:
```python
# 1. Get Embedding of Query and Key  
query_repr = model.get_query_repr(sa_seg)
key_repr = model.get_key_repr(sa_seg)

# 2. Calculate dot product
dot_product = torch.dot(query_repr.view(-1), key_repr.view(-1))

# 3. Divide by model.temperature to get the final score
similarity_score = dot_product / model.model.temperature
```


## Inference with DOMO
You can quickly run the DOMO inference code using the provided script:
```bash
python DOMO_inference.py
```

DOMO is engineered to synthesize a cohesive, multi-domain protein sequence from a diverse set of input domains. The core generation pipeline entails tokenizing the input domain list and feeding it into the model's conditional generation module to decode the integrated protein sequence:
```python
# 1. Tokenize the input list of domains
tokenized_domain = tokenizer(domain_list, return_tensors="pt", padding=True, truncation=True, max_length=512)
domain_ids = tokenized_domain.input_ids.to(device)
domain_masks = tokenized_domain.attention_mask.to(device)

# 2. Specify the number of domains per protein
num_domains_per_protein = torch.tensor([len(domain_list)]).to(device)

# 3. Generate the combined sequence
domain_comb_sequence = model.generate(
    domain_ids=domain_ids, 
    domain_masks=domain_masks, 
    num_domains_per_protein=num_domains_per_protein)["output_seqs"]
```