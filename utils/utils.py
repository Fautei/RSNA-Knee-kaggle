
from pytorch_lightning.loggers import MLFlowLogger, CSVLogger
from typing import Optional

def create_logger(cfg: dict, run_name: Optional[str]):
    mlflow_cfg = cfg["mlflow"]

    logger = MLFlowLogger(
        tracking_uri=mlflow_cfg["tracking_uri"],
        experiment_name=mlflow_cfg["experiment_name"],
        run_name=run_name
    )
    return logger
