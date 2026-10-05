from monai.transforms import (
    Compose,
    Resized,
    RandFlipd,
    NormalizeIntensityd,
    RandZoomd,
    CenterSpatialCropd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandAdjustContrastd
)

def get_train_transform(cfg) -> Compose:
    image_size = cfg["image_size"]
    crop_size = cfg["crop_size"]
    flip_prob = cfg["flip_prob"]
    zoom_prob = cfg["zoom_prob"]
    min_zoom = cfg["min_zoom"]
    max_zoom = cfg["max_zoom"]
    
    noise_prob = cfg["noise_prob"]
    noise_std = cfg["noise_std"]
    scale_prob = cfg["scale_intensity_prob"]
    scale_factors = cfg["scale_intensity_factors"]
    shift_prob = cfg["shift_intensity_prob"]
    shift_offsets = cfg["shift_intensity_offsets"]
    contrast_prob = cfg["adjust_contrast_prob"]
    contrast_gamma = cfg["adjust_contrast_gamma"]

    keys = ["image_axial", "image_sagittal", "image_coronal"]

    return Compose([
        Resized(keys=keys, spatial_size=(-1, *image_size), allow_missing_keys=True),
        RandZoomd(keys=keys, prob=zoom_prob, min_zoom=min_zoom, max_zoom=max_zoom, padding_mode="constant", allow_missing_keys=True),
        RandFlipd(keys=["image_axial", "image_coronal"], prob=flip_prob, spatial_axis=[-1], allow_missing_keys=True),
        CenterSpatialCropd(keys=keys, roi_size=(-1, *crop_size), allow_missing_keys=True),
        RandGaussianNoised(keys=keys, prob=noise_prob, mean=0.0, std=noise_std, allow_missing_keys=True),
        RandScaleIntensityd(keys=keys, prob=scale_prob, factors=scale_factors, allow_missing_keys=True),
        RandShiftIntensityd(keys=keys, prob=shift_prob, offsets=shift_offsets, allow_missing_keys=True),
        RandAdjustContrastd(keys=keys, prob=contrast_prob, gamma=contrast_gamma, allow_missing_keys=True),
        NormalizeIntensityd(keys=keys, allow_missing_keys=True),
    ])

def get_val_transform(cfg) -> Compose:
    image_size = cfg["image_size"]
    crop_size = cfg["crop_size"]
    return Compose([
        Resized(keys=["image_axial", "image_sagittal", "image_coronal"], spatial_size=(-1, *image_size), allow_missing_keys=True),
        CenterSpatialCropd(keys=["image_axial", "image_sagittal", "image_coronal"], roi_size=(-1, *crop_size), allow_missing_keys=True),
        NormalizeIntensityd(keys=["image_axial", "image_sagittal", "image_coronal"], allow_missing_keys=True),
    ])