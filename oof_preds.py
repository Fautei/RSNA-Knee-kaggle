import os
import math
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import KFold
from argparse import ArgumentParser
from typing import Optional, List, Dict
from tqdm import tqdm

# Импорт ваших модулей
from utils import create_logger, get_val_transform, get_train_transform, load_config
from datasets import RSNAKADDataset
from main import RSNAKADLightningModule, train_model
from export import export

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

def generate_oof_predictions(cfg: dict, run_name: str, output_csv_path: str):
    pl.seed_everything(int(cfg["seed"]), workers=True)

    train_studies = pd.read_csv(cfg["train_df_path"])
    val_studies = pd.read_csv(cfg["val_df_path"])

    val_uids = set(val_studies["StudyInstanceUID"].unique())
    train_studies = train_studies[
            ~train_studies["StudyInstanceUID"].isin(val_uids)
        ].reset_index(drop=True).copy()

    print(len(train_studies))
    
    train_series = pd.read_csv(cfg["train_series_path"])
    val_series = pd.read_csv(cfg["val_series_path"])

    label_columns = cfg["label_columns"]
    n_folds = cfg.get("n_folds", 3) 
    
    oof_preds = np.zeros((len(train_studies), len(label_columns)))
    
    train_studies['fold'] = -1

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=int(cfg["seed"]))
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_studies)):
        print(f"\n--- Training Fold {fold_idx + 1}/{n_folds} ---")
        
        train_studies.loc[val_idx, 'fold'] = fold_idx
        
        current_train_studies = train_studies.iloc[train_idx].reset_index(drop=True)
        current_val_studies = train_studies.iloc[val_idx].reset_index(drop=True) # Это наш OOF set

        checkpoint_dir = os.path.join("checkpoints", run_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # train model require csv
        df_path = f"{checkpoint_dir}/fold_{fold_idx}.csv"
        current_train_studies.to_csv(df_path)
        cfg["train_df_path"] = df_path

        results = train_model(cfg, run_name + f"fold_{fold_idx}")
        
        # Датасет для получения OOF предсказаний (отложенный фолд)
        oof_dataset = RSNAKADDataset(
            df_studies=current_val_studies,
            df_series=train_series, 
            label_columns=label_columns,
            series_root=cfg["train_series_dir"],
            split="val", 
            transform=get_val_transform(cfg)
        )

        oof_dataloader = torch.utils.data.DataLoader(
            oof_dataset, 
            1, 
            shuffle=False, 
            collate_fn=lambda x: x, 
            num_workers=cfg["num_workers"]
        )

        enc_cfg = cfg["encoder"]
        models = []
        checkpoints_per_class = results["best_per_class"]
        for i, label_name in enumerate(label_columns):
            best_class_model_path = checkpoints_per_class[i]
                
            print(f"Loading best model for class '{label_name}' from: {best_class_model_path}")
            
            model = RSNAKADLightningModule.load_from_checkpoint(
                best_class_model_path,
                weights_only=False,
                num_classes=len(label_columns),
                epochs=cfg["epochs"],
                lr=cfg["learning_rate"],
                encoder_name=str(enc_cfg["name"]),
                pretrained=bool(enc_cfg["pretrained"]),
                n_queries=int(enc_cfg["n_queries"]),
                num_heads=int(enc_cfg["num_heads"]),
                pooling_inner_dim=int(enc_cfg["inner_dim"])
            ).eval().to("cuda:0").half()

            models.append(model)

        print(f"\nAll {len(models)} models loaded successfully!\n")

        all_class_preds = [[] for _ in range(len(label_columns))]

        for batch in tqdm(oof_dataloader, total=len(oof_dataloader)):
            images = batch[0]["images"] 
            
            for model_idx, model in enumerate(models):
                with torch.no_grad():
                    features = []
                    for item in images:
                        image = item["image"].unsqueeze(0).to("cuda:0").half()
                        feats = model.model.features(image) 
                        features.append(feats)
                    
                    if features:
                        features = torch.cat(features, dim=2)
                    logits = model.model.logits(features)
                    
                    probs = torch.sigmoid(logits)[:, model_idx].detach().cpu().numpy()
                    all_class_preds[model_idx].append(probs)

        for i in range(len(label_columns)):
            class_preds_array = np.concatenate(all_class_preds[i], axis=0)
            oof_preds[val_idx, i] = class_preds_array

    oof_df = train_studies.copy()
    for i, col in enumerate(label_columns):
        oof_df[col] = oof_preds[:, i]
        
    output_path = f"{output_csv_path}/{run_name}.csv"
    oof_df.to_csv(output_path, index=False)
    print(f"OOF predictions saved to {output_path}")

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = ArgumentParser()
    parser.add_argument('--config_path', default="configs/default.yaml", help='Config path')
    parser.add_argument('--run_name', default="oof_run", help='Run name prefix')
    parser.add_argument('--output_csv_path', default="", help='Out folder path')
    args = parser.parse_args()

    config = load_config(args.config_path)
    generate_oof_predictions(config, args.run_name, args.output_csv_path)
