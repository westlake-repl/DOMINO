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