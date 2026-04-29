## Download model weights

We have released the DOMINO model weights on Hugging Face.

**Model repository**: [westlake-repl/DOMINO](https://huggingface.co/westlake-repl/DOMINO)

We provide a script to download the DOMINO model weights using `huggingface-cli`, as shown below. Please download all files and put them in the `weights` directory.

```bash
# If you haven't installed huggingface_hub, run: pip install -U huggingface_hub
huggingface-cli download westlake-repl/DOMINO \
                         --repo-type model \
                         --local-dir weights/


After downloading, your directory structure should look like this:
DOMINO/
├── weights/
│   ├── DOMIN/
│   ├── DOMO/
│   └── ...
├── src/
├── DOMIN_inference.py
├── DOMO_inference.py
└── ...