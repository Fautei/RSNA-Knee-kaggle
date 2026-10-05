import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List, Dict

from models import RSNAKADModel
from torchmetrics import AUROC

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from typing import Optional, List, Dict

import pandas as pd

from utils import create_logger, get_val_transform, get_train_transform, load_config, get_default_config
from datasets import RSNAKADDataset
from argparse import ArgumentParser

class RSNAKADLightningModule(pl.LightningModule):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return nn.BCEWithLogitsLoss()(logits, labels)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images = batch["images"]
        labels = batch["label"]

        logits = self(images)
        loss = self._loss(logits, labels)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        images = batch[0]["images"]
        labels = batch[0]["label"].unsqueeze(0)

        features = []
        for item in images:
            image = item["image"].unsqueeze(0)
            feats = self.model.features(image)
            features.append(feats)

        features = torch.cat(features, dim=2)

        logits = self.model.logits(features)
        loss = self._loss(logits, labels)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        probs = torch.sigmoid(logits)

        self.val_preds.append(probs.detach().cpu())
        self.val_labels.append(labels.detach().cpu().long())

        return loss


    def on_validation_epoch_end(self) -> None:
        if len(self.val_preds) == 0:
            return

        # Конкатенируем все батчи
        all_preds = torch.cat(self.val_preds, dim=0)
        all_labels = torch.cat(self.val_labels, dim=0)

        # Считаем AUROC
        auroc = self.val_auroc(all_preds, all_labels)
        
        # Логируем метрику
        self.log(
            "val_auc",
            auroc,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )

        # Очищаем списки для следующей эпохи
        self.val_preds.clear()
        self.val_labels.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.hparams.epochs),
            eta_min=1e-6,
        )

        return [optimizer], [scheduler]


class RSNAKADLightningModule2(pl.LightningModule):
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

def train_model(
    cfg: Optional[dict] = None,
    limit_train_batches: Optional[float] = None,
    limit_val_batches: Optional[float] = None,
):
    pl.seed_everything(int(cfg["seed"]), workers=True)

    val_studies = pd.read_csv(cfg["val_df_path"])
    train_studies = pd.read_csv(cfg["train_df_path"])

    val_uids = set(val_studies["StudyInstanceUID"].unique())
    train_studies = train_studies[
            ~train_studies["StudyInstanceUID"].isin(val_uids)
        ].copy()

    train_series = pd.read_csv(cfg["train_series_path"])
    val_series = pd.read_csv(cfg["val_series_path"])

    label_columns = cfg["label_columns"]
    train_series_dir = cfg["train_series_dir"]
    val_series_dir = cfg["val_series_dir"]
    batch_size = cfg["batch_size"]
    slice_count = cfg["slice_count"]

    train_dataset = RSNAKADDataset(
        df_studies=train_studies,
        df_series=train_series,
        label_columns=label_columns,
        series_root=train_series_dir,
        split="train",
        transform=get_train_transform(cfg),
        train_slices=slice_count
    )

    val_dataset = RSNAKADDataset(
        df_studies=val_studies,
        df_series=val_series,
        label_columns=label_columns,
        series_root=val_series_dir,
        split="val",
        transform=get_val_transform(cfg)
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=12)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, 1, shuffle=False, collate_fn=lambda x: x, num_workers=12)

    label_names = cfg["label_columns"]
    enc_cfg = cfg["encoder"]

    model = RSNAKADLightningModule(
        num_classes=len(label_names),
        epochs=cfg["epochs"],
        lr=cfg["learning_rate"],
        encoder_name=str(enc_cfg["name"]),
        pretrained=bool(enc_cfg["pretrained"]),
        n_queries=int(enc_cfg["n_queries"]),
        num_heads=int(enc_cfg["num_heads"]),
        pooling_inner_dim=int(enc_cfg["inner_dim"])
    )


    logger = create_logger(cfg)

    logger.log_hyperparams(cfg)

    checkpoint_dir = "checkpoints"

    chpt = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="best_{epoch}-{val_auc:.4f}",
                monitor="val_auc",
                mode="max",
                save_top_k=1,
                verbose=True
            )

    callbacks = [
        chpt, 
        LearningRateMonitor(logging_interval="epoch"),
        
    ]

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    precision = str(cfg["precision"])

    if accelerator == "cpu":
        precision = "32"

    devices = int(cfg["gpus"]) if accelerator == "gpu" else 1

    if limit_train_batches is None:
        limit_train_batches = float(cfg["limit_train_batches"])

    if limit_val_batches is None:
        limit_val_batches = float(cfg["limit_val_batches"])

    trainer = pl.Trainer(
        max_epochs=cfg["epochs"],
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=[logger],
        callbacks=callbacks,
        #val_check_interval=500,
        num_sanity_val_steps=2,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        accumulate_grad_batches=cfg["accumulate_grad_batches"],
        gradient_clip_val=1.0,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


    best_path = chpt.best_model_path
    best_metric = chpt.best_model_score
    
    print("Best checkpoint:", best_path)
    return {
        "best": best_path,
        "metric": best_metric,
        "data": cfg["pseudo_csv"],
    }

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = ArgumentParser()
    parser.add_argument('config_path',default="configs/default.yaml", help='Конфиг')

    args = parser.parse_args()

    config_path = args.config_path
    config = load_config(config_path)

    train_model(config)