
from pytorch_lightning.loggers import MLFlowLogger, CSVLogger

def create_logger(cfg: dict):
    mlflow_cfg = cfg["mlflow"]

    logger = MLFlowLogger(
        tracking_uri=mlflow_cfg["tracking_uri"],
        experiment_name=mlflow_cfg["experiment_name"],
    )
    return logger
