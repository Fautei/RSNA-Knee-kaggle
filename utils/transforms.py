from monai.transforms import (
    Compose,
    Resized,
    RandFlipd,
    NormalizeIntensityd,
    RandZoomd,
    CenterSpatialCropd
)

def get_train_transform(cfg) -> Compose:
    image_size = cfg["image_size"]
    crop_size = cfg["crop_size"]
    flip = cfg["flip_prob"]
    zoom=cfg["zoom_prob"]
    min_zoom = cfg["min_zoom"]
    max_zoom = cfg["max_zoom"]
    return Compose(
        [
            Resized(keys=["image_axial", "image_sagittal", "image_coronal"], spatial_size=(-1, *image_size), allow_missing_keys=True),
            RandZoomd(keys=["image_axial", "image_sagittal", "image_coronal"], prob=zoom, min_zoom=min_zoom, max_zoom=max_zoom, padding_mode="constant", allow_missing_keys=True),
            RandFlipd(keys=["image_axial", "image_coronal"], prob=flip, spatial_axis=[-1], allow_missing_keys=True),
            CenterSpatialCropd(keys=["image_axial", "image_sagittal", "image_coronal"], roi_size=(-1, *crop_size), allow_missing_keys=True),
            NormalizeIntensityd(keys=["image_axial", "image_sagittal", "image_coronal"], allow_missing_keys=True),
        ]
    )

def get_val_transform(cfg) -> Compose:
    image_size = cfg["image_size"]
    crop_size = cfg["crop_size"]
    return Compose(
        [
            Resized(keys=["image_axial", "image_sagittal", "image_coronal"], spatial_size=(-1, *image_size), allow_missing_keys=True),
            CenterSpatialCropd(keys=["image_axial", "image_sagittal", "image_coronal"], roi_size=(-1, *crop_size), allow_missing_keys=True),
            NormalizeIntensityd(keys=["image_axial", "image_sagittal", "image_coronal"], allow_missing_keys=True),
        ]
    )