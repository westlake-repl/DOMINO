import argparse
from omegaconf import OmegaConf
import logging
import torch 
from transformers import EsmTokenizer
import sys 
sys.path.append("./src/DOMO")
from utils.init_utils import construct_class_by_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device="cuda"
EXAMPLE_INPUTDOMAIN1 = ['MTAFQKLDFSVNDVIESVKDGNVIGRGGAGVVYHGKTPNGVEIAVKKLMGFNGINGHDHGFKAEIRTLGNIRHRNIVRLLAFCSNKDTNLLVYDYMRNGSLGEALHGKKGGILGWNLRYKIAVDAAKGLCYLHHDCEPLIVHRDVKSNNILLDSSFEARVADFGLAKFL',
                'SGNNFSGPIPPSIGQLRQVVKIDLSGNSLEARIPLEIGNCLHLNYLDLSKNELSGSIPQEISDIKILNYLNLSRNHLNDTIPQSIASMKSLTTVDFSYNDLAGKLPETGQFSVFNATSFIGNPRLCGPLLN']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/DOMO/configs/DOMO_config.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    model = construct_class_by_name(**config.Model.kwargs, logger=logger)
    assert config.Checkpoint_path is not None
    logger.info(f"Loading checkpoint from {config.Checkpoint_path}...")
    model.load_state_dict(torch.load(config.Checkpoint_path))
    model.eval()
    model = model.to(torch.device(device))
    logger.info(f"Checkpoint loaded successfully")

    tokenizer = model.tokenizer

    ## an example of 2 domains generation
    domain_list = EXAMPLE_INPUTDOMAIN1
    for idx, domain in enumerate(domain_list):
        logger.info(f"Domain {idx+1}: {domain}")
    tokenized_domain = tokenizer(domain_list, return_tensors="pt", padding=True, truncation=True, max_length=512)
    domain_ids = tokenized_domain.input_ids.to(device)
    domain_masks = tokenized_domain.attention_mask.to(device)
    num_domains_per_protein = torch.tensor([len(domain_list)]).to(device)
    domain_comb_sequence = model.generate(domain_ids=domain_ids, 
                domain_masks=domain_masks, 
                num_domains_per_protein=num_domains_per_protein)["output_seqs"][0]

    logger.info(f"Domain combined sequence: {domain_comb_sequence}")