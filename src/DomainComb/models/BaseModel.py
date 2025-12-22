import torch


class BaseModel(torch.nn.Module):
    def __init__(self, logger, **kwargs):
        super().__init__()
        self.logger = logger
        self.kwargs = kwargs
        self.logger.info(f"BaseModel initialized with kwargs: {kwargs}")

    def forward(self, batch):
        pass

    def set_objective_and_metrics(self, stage: str = "train"):
        raise NotImplementedError("set_objective_and_metrics is not implemented")
