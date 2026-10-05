import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List, Dict

from models import RSNAKADModel
from torchmetrics import AUROC
from argparse import ArgumentParser
import os

from utils import load_config

class RSNAKADWrapper(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        epochs = 10,
        lr = 1e-3,
        encoder_name: str = "convnext_tiny.fb_in1k",
        pretrained: bool = True,
        n_queries: int = 16,
        num_heads: int = 8,
        pooling_inner_dim: int = 128, 
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes

        self.model = RSNAKADModel(
            num_classes=self.num_classes,
            encoder_name=encoder_name,
            pretrained=pretrained,
            n_queries=n_queries,
            num_heads=num_heads,
            pooling_inner_dim=pooling_inner_dim
        )

        self.val_auroc = AUROC(
            task='multilabel',
            num_labels=num_classes,
            average='macro'  # или 'weighted', 'micro' в зависимости от задачи
        )

        self.val_preds: List[torch.Tensor] = []
        self.val_labels: List[torch.Tensor] = []


    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x1, x2, x3], dim=1)
        return self.model(x)


def export(ckpt_path, cfg):
    label_names = cfg["label_columns"]
    enc_cfg = cfg["encoder"]

    model = RSNAKADWrapper.load_from_checkpoint(
        ckpt_path,
        weights_only=False,
        num_classes=len(label_names),
        epochs=cfg["epochs"],
        lr=cfg["learning_rate"],
        encoder_name=str(enc_cfg["name"]),
        pretrained=bool(enc_cfg["pretrained"]),
        n_queries=int(enc_cfg["n_queries"]),
        num_heads=int(enc_cfg["num_heads"]),
        pooling_inner_dim=int(enc_cfg["inner_dim"])
    )


    checkpoint_dir = os.path.dirname(ckpt_path)
    filename = os.path.basename(ckpt_path)
    output_path = os.path.join(checkpoint_dir, f"{filename[:-5]}.torchscript")
    print(f"Saving torchscript to: {output_path}")

    slice_size = cfg["crop_size"]
    
    model.to_torchscript(output_path, method='trace', example_inputs=(torch.randn(2, 16, 1, *slice_size), torch.randn(2, 16, 1, *slice_size), torch.randn(2, 16, 1, *slice_size)))


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = ArgumentParser()
    parser.add_argument('--config_path',default="configs/default.yaml", help='Config path')
    parser.add_argument('--ckpt_path', help='Checkpoint path')

    args = parser.parse_args()

    config_path = args.config_path
    ckpt_path = args.ckpt_path
    config = load_config(config_path)

    export(ckpt_path, config)