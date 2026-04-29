## Download model weights

We have released the DOMINO model weights on Hugging Face. Click **[here](https://huggingface.co/westlake-repl/DOMINO)** to access the model repository.

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