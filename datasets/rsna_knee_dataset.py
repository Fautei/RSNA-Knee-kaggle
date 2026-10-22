import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
import pydicom
import random
from collections import defaultdict



class RSNAKADDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df_studies: pd.DataFrame,
        df_series: pd.DataFrame,
        label_columns: List[str],
        series_root: str,
        split: str,

        train_slices = 32,
        transform=None,
    ):
        self.split = str(split)
        self.label_columns = list(label_columns)
        self.series_root = str(series_root)
        self.transform = transform
        self.studies = df_studies
        self.series = df_series
        self.train_slices = train_slices

        # Increase amount of studies by flipping their laterality:
        if self.split == "val":
            studies1 = self.studies.copy()
            studies1["flip"] = 0
            studies2 = self.studies.copy()
            studies2["flip"] = 1
            self.studies = pd.concat([studies1, studies2])


    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, idx: int) -> Dict:
        study = self.studies.iloc[idx]
        study_uid = str(study["StudyInstanceUID"])
        series = self.series[self.series["StudyInstanceUID"] == study_uid]
        label = study.loc[self.label_columns].values.astype(np.float32)

        if self.split == "train":
            all_files = []

            for i, row in series.iterrows():
                series_uid = row["SeriesInstanceUID"]
                series_path = os.path.join(self.series_root, study_uid, series_uid)
                files = self._list_files_sorted_by_instance_number(series_path)

                n_files = len(files)
                plane = str.lower(row["Anatomical_Plane"])
                key = f"image_{plane}"
                important = [(i, plane, key, f) for f in files[n_files*20//100:n_files*80//100]]
                all_files.extend(important)
            if len(all_files) < self.train_slices:
                sampled = random.choices(all_files, k=self.train_slices)
            else:
                sampled = random.sample(all_files, k=self.train_slices)

            groups = defaultdict(list)
            for i, plane, key, file_path in sampled:
                groups[i].append((plane, key, file_path))
            
            # Применяем трансформы к каждой группе отдельно и конкатенируем
            transformed_volumes = []
            for i in sorted(groups.keys()):
                group_files = groups[i]
                # Все файлы в группе имеют одинаковый plane и key (из одной серии)
                plane = group_files[0][0]
                key = group_files[0][1]
                file_paths = [f[2] for f in group_files]
                
                # Загружаем все слайсы этой группы
                slices = self._load_volume_slices(file_paths)
                volume = torch.stack(slices).unsqueeze(0)  # (1, N_slices, H, W)
                
                # Применяем трансформы ко всему объему группы
                tensor = self.transform({
                    key: volume
                })[key]
                transformed_volumes.append(tensor)
            
            # Конкатенируем все трансформированные объемы
            image = torch.cat(transformed_volumes, dim=1)  # конкатенируем по оси срезов

            return {
                "images": image.permute(1, 0, 2, 3),
                "label": torch.as_tensor(label, dtype=torch.float32).reshape(-1).contiguous(),
                "study_uid": study_uid
            }

        else:
            item = {
                "images": [],
                "label": torch.as_tensor(label, dtype=torch.float32).reshape(-1).contiguous(),
                "study_uid": study_uid
            }

            for i, row in series.iterrows():
                series_uid = row["SeriesInstanceUID"]
                series_path = os.path.join(self.series_root, study_uid, series_uid)
                files = self._list_files_sorted_by_instance_number(series_path)
                image = torch.stack(self._load_volume_slices(files)).unsqueeze(0)
                plane = str.lower(row["Anatomical_Plane"])
                key = f"image_{plane}"
                tensor = self.transform({
                    key: image
                })[key]
                
                if self.split == "val":
                    if plane in ["axial", "coronal"] and study["flip"] == 1:
                        tensor = tensor.flip(dims=[3])
                    elif plane == "sagittal" and study["flip"] == 1:
                        tensor = tensor.flip(dims=[1])

                item["images"].append({
                    "series_uid":series_uid,
                    "plane":plane,
                    "image":tensor.permute(1, 0, 2, 3),
                    "fat_suppresion":row["Fat_Suppression"],
                    "fluid_sensetive":row["Fluid_Sensitive"]
                })
            return item


    def _list_files_sorted_by_instance_number(self, series_path: str) -> List[str]:
        files_raw = [
            f for f in os.listdir(series_path)
            if os.path.isfile(os.path.join(series_path, f))
        ]
        
        if not files_raw:
            return []
            
        file_info = []
        for f in files_raw:
            full_path = os.path.join(series_path, f)
            ds = pydicom.dcmread(full_path, stop_before_pixels=True)
            inst_num = getattr(ds, 'InstanceNumber', 9999)
            file_info.append((int(inst_num), f))

        file_info.sort(key=lambda x: x[0])
        
        return [os.path.join(series_path, f[1]) for f in file_info]
    
    def _load_volume_slices(self, files: List) -> List[torch.Tensor]:
        slices = []

        for f in files:
            try:
                ds = pydicom.dcmread(f, force=True)
                img = np.asarray(ds.pixel_array, dtype=np.float32)
                img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
                slices.append(torch.from_numpy(img))

            except Exception:
                print(f"Error loading slice {f}")
                    
        return slices


if __name__ == "__main__":

    df_studies = pd.read_csv("/home/maxim/Desktop/RSNA KAD/extra_data/soft_labels_2.csv", dtype={"StudyInstanceUID": str})
    df_series = pd.read_csv("/home/maxim/Desktop/RSNA KAD/extra_data/train_series_2.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID":str})

    label_columns = [
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

    from monai.transforms import (
        Compose,
        Resized,
        RandFlipd,
        NormalizeIntensityd,
        RandZoomd,
        CenterSpatialCropd
    )

    def get_train_transform() -> Compose:
        return Compose(
            [
                Resized(keys=["image_axial", "image_sagittal", "image_coronal"], spatial_size=(-1, 384, 384), allow_missing_keys=True),
                RandZoomd(keys=["image_axial", "image_sagittal", "image_coronal"], prob=0.2, min_zoom=0.8, max_zoom=1.2, padding_mode="constant", allow_missing_keys=True),
                RandFlipd(keys=["image_axial", "image_coronal"], prob=0.5, spatial_axis=[2], allow_missing_keys=True),
                CenterSpatialCropd(keys=["image_axial", "image_sagittal", "image_coronal"], roi_size=(-1, 256, 256), allow_missing_keys=True),
                NormalizeIntensityd(keys=["image_axial", "image_sagittal", "image_coronal"], allow_missing_keys=True),
            ]
        )

    dataset = RSNAKADDataset(
        df_studies=df_studies,
        df_series=df_series,
        label_columns=label_columns,
        series_root="/home/maxim/Desktop/RSNA KAD/extra_data/downloads2",
        split="train",
        transform=get_train_transform()
    )

    sample = dataset[0]

    dataloader = torch.utils.data.DataLoader(dataset, 8, shuffle=True)
    print(next(iter(dataloader))["images"].shape)

    val_dataset = RSNAKADDataset(
        df_studies=df_studies,
        df_series=df_series,
        label_columns=label_columns,
        series_root="/home/maxim/Desktop/RSNA KAD/data/train_series",
        split="val",
        transform=get_train_transform()
    )
    valdataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False, collate_fn=lambda x: x)