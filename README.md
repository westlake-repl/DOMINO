## Download model weights

We have released the DOMINO model weights on Hugging Face.

**Model repository**: [https://huggingface.co/westlake-repl/DOMINO](https://huggingface.co/westlake-repl/DOMINO)

We provide a script to download the DOMINO model weights, as shown below. Please download all files and put them in the weights directory, e.g., weights/DOMINO/...

```bash
huggingface-cli download westlake-repl/DOMINO \
                         --repo-type model \
                         --local-dir weights/


After downloading, your directory structure should look like this:
DOMINO/
├── weights/
│   ├── DOMIN.pt
│   └── ...
├── src/
├── DOMIN_inference.py
├── DOMO_inference.py
└── ...







## Download model weights
...

## Step 2 Domain Combination

### Environment Setup
 
`pip install -r reqs_domaincomb.txt`

### Checkpoints Download

`cp -r /yuanfajie/DomainReComb/checkpoints .`

### Example Usage

`python domaincomb.py --config src/DomainComb/configs/01-AR-esm2-qwen3-200M.yaml`
