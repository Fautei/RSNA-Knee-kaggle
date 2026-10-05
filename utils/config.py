import yaml
import os



DATA_ROOT = "data"

BATCH_SIZE = 8
NUM_EPOCHS = 15
LIMIT_TRAIN_BATCHES = 1.0
LIMIT_VAL_BATCHES = 1.0
PRECISION = "16-mixed"

SLICE_COUNT = 32
IMAGE_SIZE = (384, 384)
CROP_SIZE = (256, 256)

# Encoder для 2.5D
ENCODER_NAME = "efficientnet_b3.ra2_in1k"  # можно base/large, но тяжелее
ENCODER_PRETRAINED = True
N_QUERIES = 16
NUM_HEADS = 8
INNER_DIM = 128

LABEL_COLUMNS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

CONFIG = dict(
    seed=42,
    data_root=DATA_ROOT,
    label_columns=LABEL_COLUMNS,
    batch_size=BATCH_SIZE,
    num_workers=12,
    epochs=NUM_EPOCHS,
    learning_rate=3e-4,
    weight_decay=1e-2,
    precision=PRECISION,
    gpus=1,
    limit_train_batches=LIMIT_TRAIN_BATCHES,
    limit_val_batches=LIMIT_VAL_BATCHES,
    accumulate_grad_batches=2,

    slice_count=SLICE_COUNT,
    image_size=IMAGE_SIZE,
    crop_size=CROP_SIZE,
    zoom_prob=0.5,
    flip_prob=0.5,
    min_zoom=0.8,
    max_zoom=1.2,

    noise_prob=0.0,
    noise_std=0.33,
    scale_intensity_prob=0.0,
    scale_intensity_factors=0.1,
    shift_intensity_prob=0.0,
    shift_intensity_offsets=0.1,
    adjust_contrast_prob=0.0,
    adjust_contrast_gamma=(0.7, 1.5),

    train_df_path="/home/maxim/Desktop/RSNA KAD/data/refined_predictions_advanced.csv",
    val_df_path="/home/maxim/Desktop/RSNA KAD/data/valid.csv",
    train_series_path="/home/maxim/Desktop/RSNA KAD/data/train_series.csv",
    val_series_path="/home/maxim/Desktop/RSNA KAD/data/train_series.csv",
    train_series_dir="/home/maxim/Desktop/RSNA KAD/data/train_series",
    val_series_dir="/home/maxim/Desktop/RSNA KAD/data/train_series",

    encoder=dict(
        name=ENCODER_NAME,
        pretrained=ENCODER_PRETRAINED,
        n_queries=N_QUERIES,
        num_heads=NUM_HEADS,
        inner_dim=INNER_DIM
    ),
    mlflow=dict(
        tracking_uri="mlruns",
        experiment_name="rsna_kad",
    ),
)

def get_default_config():
    return CONFIG

def load_config(path):
    print(f"Loading config from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    with open("configs/default.yaml", "w") as f:
        yaml.safe_dump(CONFIG, f)
