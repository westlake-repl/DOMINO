import argparse
import torch
import logging
from omegaconf import OmegaConf
from src.DOMIN.models.ted.ted_domain_model import TedDomainModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/DOMIN/configs/DOMIN_config.yaml")
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

        logger.info(f"Query Shape:  {query_repr.shape}")
        logger.info(f"Key Shape:    {key_repr.shape}")
        temp_val = model.temperature.item() if torch.is_tensor(model.temperature) else model.temperature
        logger.info(f"Temperature:  {temp_val}")
        logger.info(f"Similarity score (self): {similarity_score.item()}")