import argparse
import torch
import os
from omegaconf import OmegaConf
from src.DomainSearch.models.ted.ted_domain_model import TedDomainModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ted_search.yaml")
    args = parser.parse_args()

    # Loading config
    conf = OmegaConf.load(args.config)

    model_config = {
        "config_path": conf.Model.kwargs.config_path,
        "from_checkpoint": conf.Checkpoint_path
    }
    
    device = "cuda"
    model = TedDomainModel(**model_config)
    model.to(device).eval()

    # test sequence
    sa_seg = "TcGvDvRvKdRdEdFdLfEdLwGdRcKdAqGvRdFpPpAwAiSdTgSpNvGgEiIfSgIeWf<unk>ElEsRnRvRqPlAlEvNqAlRvLlTlHvGvLlLcRvEvRlDlIfPdVfLsSdDsRnShPsIwVtPwVgLwVqGqEaDdRvMlClKvRqMlSqAvLqPcLcEvRpHvGsAyYhVwQdAwIdDdApPpSnVdPdArGrErEiItLgRtIgArPrShAsVsHdEdTpEvEnIsHvRvFsVsDvAsLsDsGvIsWcSvEvLsGv"

    with torch.no_grad():
        query_repr = model.get_query_repr(sa_seg)
        key_repr = model.get_key_repr(sa_seg)
        
        print(f"--- Search Results ---")
        print(f"Model Folder: {model_config['config_path']}")
        print(f"Query Shape: {query_repr.shape}")
        print(f"Key Shape:   {key_repr.shape}")
