import sys
import os
import glob
import warnings
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSplitter, QTableWidget, QTableWidgetItem, 
                             QLabel, QSlider, QGroupBox, QTextEdit, QHeaderView, QPushButton,
                             QFileDialog, QMessageBox, QTabWidget, QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import pydicom
from PIL import Image
import SimpleITK as sitk
import pyvista as pv
from pyvistaqt import QtInteractor

warnings.filterwarnings("ignore", message=".*Non uniform sampling.*")

PATIENT_DATA_CSV = os.getenv("PATIENT_DATA_CSV", "data/valid.csv")
SERIES_DATA_CSV = os.getenv("SERIES_DATA_CSV", "data/train_series.csv")
BASE_DIR = os.getenv("BASE_DIR", "data/train_series")

# Целевое изотропное разрешение для объединения (мм)
TARGET_SPACING = (1.0, 1.0, 1.0)

class DicomViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Просмотр МРТ коленного сустава + 3D")
        self.setGeometry(100, 100, 1400, 900)

        try:
            self.df_patients = pd.read_csv(PATIENT_DATA_CSV)
            self.df_series = pd.read_csv(SERIES_DATA_CSV)
        except Exception as e:
            print(f"Ошибка загрузки CSV: {e}")
            self.df_patients = pd.DataFrame()
            self.df_series = pd.DataFrame()
        
        self.current_study_uid = None
        self.current_series_uid = None
        self.current_slices = [] 
        
        # Два финальных 3D объема на единой сетке
        self.volume_normal = None
        self.volume_fs = None

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        view_area = QWidget()
        view_layout = QVBoxLayout(view_area)
        
        toolbar = QHBoxLayout()
        self.export_btn = QPushButton("Экспорт серии в JPEG")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_series_to_jpeg)
        toolbar.addWidget(self.export_btn)
        toolbar.addStretch()
        view_layout.addLayout(toolbar)

        self.main_tabs = QTabWidget()
        
        # === Вкладка 1: 2D Просмотр ===
        tab_2d = QWidget()
        layout_2d = QVBoxLayout(tab_2d)
        self.image_label = QLabel("Выберите серию для просмотра")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; color: white; min-height: 400px;")
        layout_2d.addWidget(self.image_label)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.on_slider_changed)
        layout_2d.addWidget(self.slider)
        
        self.main_tabs.addTab(tab_2d, "2D Срезы")

        # === Вкладка 2: 3D Реконструкция ===
        tab_3d = QWidget()
        layout_3d = QVBoxLayout(tab_3d)
        
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Объем:"))
        self.render_mode_combo = QComboBox()
        self.render_mode_combo.addItems(["Normal", "Fat Suppression"])
        self.render_mode_combo.currentIndexChanged.connect(self._update_3d_render)
        mode_layout.addWidget(self.render_mode_combo)
        mode_layout.addStretch()
        layout_3d.addLayout(mode_layout)
        
        self.pv_widget = QtInteractor(tab_3d)
        layout_3d.addWidget(self.pv_widget)
        
        self.main_tabs.addTab(tab_3d, "3D Реконструкция")
        view_layout.addWidget(self.main_tabs)

        # --- Правая часть ---
        info_panel = QWidget()
        info_layout = QVBoxLayout(info_panel)
        info_panel.setFixedWidth(500)

        patient_group = QGroupBox("Пациенты (Studies)")
        p_layout = QVBoxLayout()
        self.patient_table = QTableWidget()
        self.populate_patient_table()
        self.patient_table.cellClicked.connect(self.on_patient_selected)
        p_layout.addWidget(self.patient_table)
        patient_group.setLayout(p_layout)
        info_layout.addWidget(patient_group)

        series_group = QGroupBox("Серии (Series)")
        s_layout = QVBoxLayout()
        self.series_table = QTableWidget()
        self.series_table.setColumnCount(5)
        self.series_table.setHorizontalHeaderLabels(["Study UID", "Series UID", "Fluid", "Fat Sup", "Plane"])
        self.series_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.series_table.cellClicked.connect(self.on_series_selected)
        s_layout.addWidget(self.series_table)
        series_group.setLayout(s_layout)
        info_layout.addWidget(series_group)

        meta_group = QGroupBox("Метаданные текущего снимка")
        m_layout = QVBoxLayout()
        self.meta_text = QTextEdit()
        self.meta_text.setReadOnly(True)
        m_layout.addWidget(self.meta_text)
        meta_group.setLayout(m_layout)
        info_layout.addWidget(meta_group)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(view_area)
        splitter.addWidget(info_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

    # ==================== ТАБЛИЦЫ И НАВИГАЦИЯ ====================
    def populate_patient_table(self):
        if self.df_patients.empty: return
        self.patient_table.setColumnCount(len(self.df_patients.columns))
        self.patient_table.setHorizontalHeaderLabels(self.df_patients.columns)
        self.patient_table.setRowCount(len(self.df_patients))
        for i, row in self.df_patients.iterrows():
            for j, value in enumerate(row):
                self.patient_table.setItem(i, j, QTableWidgetItem(str(value)))

    def on_patient_selected(self, row, column):
        self.current_study_uid = self.df_patients.iloc[row]['StudyInstanceUID']
        filtered = self.df_series[self.df_series['StudyInstanceUID'] == self.current_study_uid]
        self.series_table.setRowCount(len(filtered))
        for i, (_, r) in enumerate(filtered.iterrows()):
            for j, v in enumerate(r):
                self.series_table.setItem(i, j, QTableWidgetItem(str(v)))
        
        # Строим оба 3D объема при выборе пациента
        self._build_all_volumes()

    def on_series_selected(self, row, column):
        series_uid = self.series_table.item(row, 1).text()
        self.current_series_uid = series_uid
        self.load_dicom_series(series_uid)

    # ==================== ЗАГРУЗКА И 2D ПРОСМОТР ====================
    def load_dicom_series(self, series_uid):
        if not self.current_study_uid: return
        series_path = os.path.join(BASE_DIR, self.current_study_uid, series_uid)
        
        if not os.path.exists(series_path):
            self.image_label.setText(f"Папка не найдена: {series_path}")
            self.export_btn.setEnabled(False)
            return

        dcm_files = glob.glob(os.path.join(series_path, "*.dcm"))
        if not dcm_files:
            self.image_label.setText("DICOM файлы не найдены")
            self.export_btn.setEnabled(False)
            return

        try:
            slices_with_numbers = []
            for f in dcm_files:
                try:
                    ds = pydicom.dcmread(f, stop_before_pixels=True)
                    slices_with_numbers.append((ds.InstanceNumber, f))
                except Exception:
                    continue
            slices_with_numbers.sort(key=lambda x: x[0])
            self.current_slices = [path for _, path in slices_with_numbers]
        except Exception:
            self.current_slices = sorted(dcm_files)

        if not self.current_slices:
            self.image_label.setText("Не удалось прочитать метаданные")
            self.export_btn.setEnabled(False)
            return

        self.slider.setMaximum(len(self.current_slices) - 1)
        self.slider.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.slider.setValue(0)
        self.display_slice(0)

    def on_slider_changed(self, value):
        self.display_slice(value)

    def display_slice(self, index):
        if not self.current_slices or index >= len(self.current_slices): return
        try:
            ds = pydicom.dcmread(self.current_slices[index])
            normalized = self.normalize_pixel_data(ds.pixel_array)
            h, w = normalized.shape
            q_img = QImage(normalized.data, w, h, w, QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(q_img).scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pix)
            self.update_metadata(ds)
        except Exception as e:
            self.image_label.setText(f"Ошибка: {str(e)}")

    def normalize_pixel_data(self, pixel_array):
        if pixel_array.dtype == np.uint8: return pixel_array
        mn, mx = pixel_array.min(), pixel_array.max()
        if mx == mn: return np.zeros_like(pixel_array, dtype=np.uint8)
        return ((pixel_array - mn) / (mx - mn) * 255).astype(np.uint8)

    def update_metadata(self, ds):
        meta_lines = []
        excluded = {'PixelData', 'OverlayData', 'CurveData'}
        for elem in ds:
            if elem.keyword in excluded: continue
            if elem.VR == 'OB' and len(elem.value) > 1024:
                val = f"[Binary Data: {len(elem.value)} bytes]"
            else:
                val = str(elem.value)
            tag = f"({elem.tag.group:04X},{elem.tag.element:04X})"
            kw = elem.keyword if elem.keyword else "Unknown"
            meta_lines.append(f"{tag} {kw}: {val}")
        self.meta_text.setText("\n".join(meta_lines))

    # ==================== КОРРЕКТНАЯ МУЛЬТИСЕРИЙНАЯ 3D РЕКОНСТРУКЦИЯ ====================
    def _build_all_volumes(self):
        """
        Корректная сборка: каждая серия → отдельный объем → ресемплинг на единую сетку → слияние
        """
        if not self.current_study_uid:
            return
        
        patient_series = self.df_series[self.df_series['StudyInstanceUID'] == self.current_study_uid]
        
        normal_images = []
        fs_images = []
        
        # Шаг 1: Читаем каждую серию как ОТДЕЛЬНЫЙ объем с правильной геометрией
        for _, row in patient_series.iterrows():
            series_uid = row['SeriesInstanceUID']
            is_fs = int(row['Fat_Suppression']) == 1
            series_path = os.path.join(BASE_DIR, self.current_study_uid, series_uid)
            
            if not os.path.exists(series_path):
                continue
            
            img = self._read_single_series(series_path)
            if img is not None:
                print(f"  Series {series_uid[:20]}... | Plane: {row['Anatomical_Plane']} | "
                      f"FS: {is_fs} | Size: {img.GetSize()} | Spacing: {tuple(round(s,2) for s in img.GetSpacing())}")
                if is_fs:
                    fs_images.append(img)
                else:
                    normal_images.append(img)
        
        # Шаг 2: Ресемплируем и объединяем в единый объем
        self.volume_normal = self._merge_volumes(normal_images) if normal_images else None
        self.volume_fs = self._merge_volumes(fs_images) if fs_images else None
        
        print(f"\nFinal Normal volume: {self.volume_normal.shape if self.volume_normal is not None else 'None'}")
        print(f"Final FS volume: {self.volume_fs.shape if self.volume_fs is not None else 'None'}")
        
        self.main_tabs.setCurrentIndex(1)
        self._update_3d_render()

    def _read_single_series(self, series_path):
        """Читает ОДНУ серию как отдельный 3D объем с корректной геометрией"""
        try:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(series_path)
            if not dicom_names:
                return None
            
            reader.SetFileNames(dicom_names)
            reader.MetaDataDictionaryArrayUpdateOn()
            image = reader.Execute()
            return image
        except Exception as e:
            print(f"  Ошибка чтения серии {series_path}: {e}")
            return None

    def _merge_volumes(self, sitk_images):
        """
        Ресемплирует на единую изотропную сетку, усредняет перекрывающиеся области,
        затем нормализует результат к диапазону [0, 1] для стабильной визуализации.
        """
        if not sitk_images:
            return None
        
        # === 1. Вычисляем общий bounding box ===
        all_corners = []
        for img in sitk_images:
            size = img.GetSize()
            for corner_idx in range(8):
                idx = [
                    (corner_idx & 1) * (size[0] - 1),
                    (corner_idx & 2) // 2 * (size[1] - 1),
                    (corner_idx & 4) // 4 * (size[2] - 1)
                ]
                phys_point = img.TransformIndexToPhysicalPoint(idx)
                all_corners.append(phys_point)
        
        all_corners = np.array(all_corners)
        min_corner = all_corners.min(axis=0)
        max_corner = all_corners.max(axis=0)
        
        ref_spacing = TARGET_SPACING
        new_size = [
            max(1, int(np.ceil((max_corner[i] - min_corner[i]) / ref_spacing[i])))
            for i in range(3)
        ]
        
        ref_image = sitk.Image(new_size, sitk.sitkFloat32)
        ref_image.SetSpacing(ref_spacing)
        ref_image.SetOrigin(min_corner.tolist())
        ref_image.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        
        # === 2. Ресемплируем и усредняем ===
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        
        combined = sitk.Cast(ref_image, sitk.sitkFloat32) * 0.0
        count = sitk.Cast(ref_image, sitk.sitkFloat32) * 0.0
        
        for img in sitk_images:
            resampled = resampler.Execute(img)
            float_resampled = sitk.Cast(resampled, sitk.sitkFloat32)
            mask = sitk.Cast(float_resampled != 0, sitk.sitkFloat32)
            combined += float_resampled
            count += mask
        
        count = sitk.Maximum(count, 1.0)
        averaged = combined / count
        
        # === 3. Нормализация ПОСЛЕ усреднения ===
        # Используем робастную нормализацию по перцентилям,
        # чтобы выбросы не сжимали динамический диапазон
        avg_array = sitk.GetArrayFromImage(averaged)
        
        p_low = np.percentile(avg_array[avg_array > 0], 1) if np.any(avg_array > 0) else 0
        p_high = np.percentile(avg_array[avg_array > 0], 99) if np.any(avg_array > 0) else 1
        
        if p_high - p_low < 1e-6:
            p_high = p_low + 1.0
        
        normalized = np.clip((avg_array - p_low) / (p_high - p_low), 0.0, 1.0).astype(np.float32)
        
        print(f"  Resampled → Averaged ({len(sitk_images)} series) → Normalized [0,1]")
        print(f"  Shape: {normalized.shape} | Range: [{normalized.min():.3f}, {normalized.max():.3f}]")
        
        return normalized

    def _update_3d_render(self):
        mode = self.render_mode_combo.currentText()
        vol = self.volume_fs if mode == "Fat Suppression" else self.volume_normal
            
        if vol is None:
            self.pv_widget.clear()
            self.pv_widget.background_color = "black"
            return
        
        if mode == "Fat Suppression":
            opacity = [0, 0, 0.1, 0.3, 0.6, 0.8]
            cmap = "bone"
        else:
            opacity = [0, 0, 0.05, 0.2, 0.5, 0.7]
            cmap = "gray"
        
        self.pv_widget.clear()
        
        # Для изотропного объема spacing = (1,1,1), поэтому dimensions достаточно
        if hasattr(pv, 'ImageData'):
            grid = pv.ImageData(dimensions=np.array(vol.shape) + 1)
        else:
            grid = pv.UniformGrid(dimensions=np.array(vol.shape) + 1)
        
        grid.cell_data["values"] = vol.flatten(order="F")
        
        self.pv_widget.add_volume(
            grid, scalars="values", opacity=opacity,
            cmap=cmap, shade=False, show_scalar_bar=False
        )
        self.pv_widget.view_isometric()
        self.pv_widget.background_color = "black"

    # ==================== ЭКСПОРТ ====================
    def export_series_to_jpeg(self):
        if not self.current_slices:
            QMessageBox.warning(self, "Ошибка", "Нет загруженной серии.")
            return
        save_dir = QFileDialog.getExistingDirectory(self, "Сохранить JPEG")
        if not save_dir: return
        count = 0
        try:
            for p in self.current_slices:
                ds = pydicom.dcmread(p)
                img = Image.fromarray(self.normalize_pixel_data(ds.pixel_array), 'L')
                inst = ds.InstanceNumber if hasattr(ds, 'InstanceNumber') else count
                img.save(os.path.join(save_dir, f"slice_{inst}.jpg"), "JPEG")
                count += 1
            QMessageBox.information(self, "Успех", f"Экспортировано {count} файлов")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DicomViewer()
    window.show()
    sys.exit(app.exec_())