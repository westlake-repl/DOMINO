### Environment setup

We provide a `requirement.yml` file to help you easily set up the environment. Please run the following script to create and activate the conda environment:

```bash
conda env create -f requirement.yml
conda activate DOMINO_env
```

### Download model weights

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

### Inference with DOMIN
You can quickly run the DOMIN inference code using the provided script:
```bash
python DOMIN_inference.py
```
Understanding the core logic:
The script demonstrates how to extract representations for a given structural sequence (sa_seg) and calculate the matching similarity score. Here is the core snippet:
```bash
# 1. Get Embedding of Query and Key  
query_repr = model.get_query_repr(sa_seg)
key_repr = model.get_key_repr(sa_seg)

# 2. Calculate dot product
dot_product = torch.dot(query_repr.view(-1), key_repr.view(-1))

# 3. Divide by model.temperature to get the final score
similarity_score = dot_product / model.model.temperature
```

### Inference with DOMO