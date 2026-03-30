import argparse
import torch
from omegaconf import OmegaConf
from src.DomainSearch.models.ted.ted_domain_model import TedDomainModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/DomainSearch/configs/TED-650M-plddt70.yaml")
    args = parser.parse_args()

    # Loading config
    conf = OmegaConf.load(args.config)

    model_config = {
        "config_path": conf.Model.kwargs.config_path,
        "from_checkpoint": conf.Checkpoint_path
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TedDomainModel(**model_config)
    model.to(device).eval()

    # test sequence
    sa_seg = "TcGvDvRvKdRdEdFdLfEdLwGdRcKdAqGvRdFpPpAwAiSdTgSpNvGgEiIfSgIeWf<unk>ElEsRnRvRqPlAlEvNqAlRvLlTlHvGvLlLcRvEvRlDlIfPdVfLsSdDsRnShPsIwVtPwVgLwVqGqEaDdRvMlClKvRqMlSqAvLqPcLcEvRpHvGsAyYhVwQdAwIdDdApPpSnVdPdArGrErEiItLgRtIgArPrShAsVsHdEdTpEvEnIsHvRvFsVsDvAsLsDsGvIsWcSvEvLsGv"

    with torch.no_grad():
        # get Embedding of Query and Key  
        query_repr = model.get_query_repr(sa_seg)
        key_repr = model.get_key_repr(sa_seg)
        
        # calculate dot product
        dot_product = torch.dot(query_repr.view(-1), key_repr.view(-1))
        
        # divide by model.temperature
        similarity_score = dot_product / model.temperature
        
        print(f"--- Search Results ---")
        print(f"Model Folder: {model_config['config_path']}")
        print(f"Query Shape:  {query_repr.shape}")
        print(f"Key Shape:    {key_repr.shape}")
        print(f"Temperature:  {model.temperature.item() if torch.is_tensor(model.temperature) else model.temperature}")
        print(f"Similarity score (self): {similarity_score.item()}")