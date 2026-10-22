from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import requests
from tqdm import tqdm

# --- Configuration ---

API_URL = "http://127.0.0.1:8080/"

# Порядок ключей должен совпадать с ожидаемым порядком в downstream задачах
KEYS = [
    "acl", "mcl", "medial_meniscus", "lateral_meniscus",
    "medial_oa", "lateral_oa", "pf_oa", "effusion",
    "synovitis", "bakers", "contusion", "fracture",
]

# Список для отображения в промпте
FINDINGS_DISPLAY = {
    "acl": "ACL tear/injury",
    "mcl": "MCL tear/injury",
    "medial_meniscus": "Medial Meniscus tear/degeneration",
    "lateral_meniscus": "Lateral Meniscus tear/degeneration",
    "medial_oa": "Medial Osteoarthritis",
    "lateral_oa": "Lateral Osteoarthritis",
    "pf_oa": "Patellofemoral Osteoarthritis",
    "effusion": "Joint Effusion",
    "synovitis": "Synovitis",
    "bakers": "Baker's Cyst",
    "contusion": "Bone Contusion/Bruise",
    "fracture": "Fracture",
}


class CheckpointManager:
    """Управляет сохранением и восстановлением прогресса псевдо-лейблинга."""
    
    def __init__(self, output_csv: str):
        self.output_csv = Path(output_csv)
        self.processed_uids: set[str] = set()
        self.existing_results: list[dict] = []
        
    def load_checkpoint(self) -> bool:
        """
        Загружает существующие результаты из CSV.
        Возвращает True если найден существующий файл с данными.
        """
        if not self.output_csv.exists():
            return False
            
        try:
            df_existing = pd.read_csv(self.output_csv)
            if df_existing.empty:
                return False
                
            # Извлекаем уже обработанные UID
            if "StudyInstanceUID" in df_existing.columns:
                self.processed_uids = set(df_existing["StudyInstanceUID"].dropna().astype(str))
                self.existing_results = df_existing.to_dict('records')
                print(f"✓ Loaded checkpoint: {len(self.processed_uids)} records from {self.output_csv}")
                return True
            else:
                print(f"⚠ Warning: 'StudyInstanceUID' column not found in {self.output_csv}")
                return False
                
        except Exception as e:
            print(f"⚠ Warning: Could not load checkpoint from {self.output_csv}: {e}")
            return False
    
    def save_batch(self, new_results: list[dict], mode: str = 'append'):
        """
        Сохраняет новые результаты в CSV.
        mode: 'append' - добавляет к существующим, 'overwrite' - перезаписывает
        """
        if not new_results:
            return
            
        df_new = pd.DataFrame(new_results)
        
        if mode == 'append' and self.output_csv.exists():
            # Добавляем к существующему файлу
            df_new.to_csv(self.output_csv, mode='a', header=False, index=False)
        else:
            # Создаем новый файл или перезаписываем
            cols = ["StudyInstanceUID"] + KEYS + ["processing_time_sec"]
            existing_cols = [c for c in cols if c in df_new.columns]
            df_new[existing_cols].to_csv(self.output_csv, index=False)
    
    def get_remaining_items(self, items: list[tuple]) -> list[tuple]:
        """Фильтрует элементы, исключая уже обработанные UID."""
        remaining = []
        skipped = 0
        
        for uid, report in items:
            uid_str = str(uid)
            if uid_str in self.processed_uids:
                skipped += 1
            else:
                remaining.append((uid, report))
        
        if skipped > 0:
            print(f"⊘ Skipped {skipped} already processed items")
        
        return remaining
    
    def merge_and_save_final(self, all_results: list[dict]):
        """Объединяет старые и новые результаты, сортирует и сохраняет финальный CSV."""
        # Объединяем существующие и новые результаты
        combined = self.existing_results + all_results
        
        if not combined:
            print("⚠ No results to save")
            return
        
        df_combined = pd.DataFrame(combined)
        
        # Убираем дубликаты по StudyInstanceUID (оставляем последние)
        if "StudyInstanceUID" in df_combined.columns:
            df_combined = df_combined.drop_duplicates(subset=["StudyInstanceUID"], keep='last')
        
        # Сортируем по UID для консистентности
        if "StudyInstanceUID" in df_combined.columns:
            df_combined = df_combined.sort_values("StudyInstanceUID").reset_index(drop=True)
        
        # Выбираем нужные колонки в правильном порядке
        cols = ["StudyInstanceUID"] + KEYS + ["processing_time_sec"]
        existing_cols = [c for c in cols if c in df_combined.columns]
        df_combined = df_combined[existing_cols]
        
        # Сохраняем
        df_combined.to_csv(self.output_csv, index=False)
        print(f"✓ Final results saved to {self.output_csv} ({len(df_combined)} records)")


import json
import time
import requests
from typing import List, Dict, Any, Optional

class LocalApiLabeler:
    def __init__(self, api_url: str = "http://127.0.0.1:8080", timeout: int = 600):
        # Используем базовый URL, так как будем менять эндпоинты
        self.base_url = api_url.rstrip('/')
        self.timeout = timeout

    def _call_chat_api(self, messages: List[Dict]) -> Optional[str]:
        """
        Использует endpoint /v1/chat/completions или /chat, который лучше 
        обрабатывает шаблоны диалогов для таких моделей как Qwen.
        """
        # Пробуем стандартный OpenAI-compatible endpoint, который есть в новых llama.cpp
        url = f"{self.base_url}/v1/chat/completions"
        
        payload = {
            "messages": messages,
            "model": "default", # Игнорируется llama.cpp, но требуется форматом
            "temperature": 0.01,
            "top_p": 0.95,
            "max_tokens": 65536, # JSON короткий
            "stop": ["</s>", "<|im_end|>"], # Стоп-токены для Qwen
            "stream": False,
            "enable_thinking": False
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            # Извлекаем контент из стандартного ответа chat completion
            return data['choices'][0]['message']['content']
        except Exception as e:
            # Если v1 endpoint не работает, пробуем старый /chat endpoint llama.cpp
            print(f"Warning: v1 chat endpoint failed ({e})")
            raise RuntimeError(f"Warning: v1 chat endpoint failed ({e})")

    def _build_messages(self, report_text: str) -> List[Dict]:
        """Формирует сообщения в формате ChatML / Qwen"""
        
        system_content = """You are an expert musculoskeletal radiologist assistant.
Your task is to extract specific findings from a knee MRI report text.

For each of the 12 findings listed below, assign a numerical score based STRICTLY on the text provided:
- 1.0 if the finding is explicitly PRESENT, abnormal, torn, or degenerated.
- 0.0 if the finding is explicitly ABSENT, normal, intact, preserved, or healthy.
- -1.0 if the finding is NOT MENTIONED, ambiguous, or if there is insufficient information to determine its status.

Important Rules:
1. Do not infer. If a structure is not mentioned, use -1.0.
2. "Normal", "intact", "preserved" = 0.0.
3. Output ONLY a valid JSON object. No markdown formatting, no explanations.

The JSON keys must be exactly these:
["acl", "mcl", "medial_meniscus", "lateral_meniscus", "medial_oa", "lateral_oa", "pf_oa", "effusion", "synovitis", "bakers", "contusion", "fracture"]

Example Output:
{"acl": 1.0, "mcl": 0.0, "medial_meniscus": -1.0, ...}
"""

        user_content = f"""Here is the MRI Report:
\"\"\"
{report_text.strip()}
\"\"\"

Provide the JSON output:"""

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    def _parse_json_response(self, content: str) -> Dict[str, float]:
        """Robust JSON parsing"""
        if not content:
            return {k: -1.0 for k in KEYS}
            
        # Очистка от markdown
        clean_content = content.replace("```json", "").replace("```", "").strip()
        
        # Поиск JSON объекта
        start_idx = clean_content.find('{')
        end_idx = clean_content.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = clean_content[start_idx : end_idx + 1]
        else:
            json_str = clean_content
            
        try:
            data = json.loads(json_str)
            result = {}
            for key in KEYS:
                val = data.get(key, -1.0)
                if isinstance(val, bool): val = 1.0 if val else 0.0
                elif isinstance(val, (int, float)): val = float(val)
                elif isinstance(val, str):
                    v_lower = val.lower().strip()
                    if v_lower in ['yes', 'present', 'true']: val = 1.0
                    elif v_lower in ['no', 'absent', 'false', 'normal']: val = 0.0
                    else: val = -1.0
                else: val = -1.0
                
                # Clamp
                if val > 1.0: val = 1.0
                if val < -1.0: val = -1.0
                result[key] = val
            return result
        except json.JSONDecodeError:
            print(f"JSON Parse Error. Content snippet: '{content[:100]}'")
            return {k: -1.0 for k in KEYS}

    def label_single(self, uid: str, report_text: str) -> Dict[str, Any]:
        start_time = time.time()
        
        messages = self._build_messages(report_text)
        content = self._call_chat_api(messages)
        
        scores = self._parse_json_response(content)
        elapsed = time.time() - start_time
        
        return {
            "StudyInstanceUID": str(uid),  # Гарантируем строковый тип
            **scores,
            "processing_time_sec": round(elapsed, 2)
        }

    def label_batch(self, items: List[tuple], output_csv: str, batch_size: int = 10):
        """
        Обрабатывает батч элементов с чекпоинтингом.
        
        Args:
            items: список кортежей (uid, report_text)
            output_csv: путь к выходному CSV файлу
            batch_size: размер батча для инкрементального сохранения
        """
        # Инициализируем менеджер чекпоинтов
        checkpoint = CheckpointManager(output_csv)
        
        # Загружаем существующий прогресс
        has_checkpoint = checkpoint.load_checkpoint()
        
        # Фильтруем уже обработанные элементы
        remaining_items = checkpoint.get_remaining_items(items)
        
        if not remaining_items:
            print("✓ All items already processed. Nothing to do.")
            # Все равно сохраняем финальный файл для консистентности
            checkpoint.merge_and_save_final([])
            return pd.DataFrame()
        
        total_remaining = len(remaining_items)
        total_original = len(items)
        already_done = total_original - total_remaining
        
        print(f"Starting pseudo-labeling for {total_remaining} remaining items "
              f"(out of {total_original} total) via {self.base_url}...")
        
        results_batch = []
        all_new_results = []
        
        for i, (uid, report) in tqdm(enumerate(remaining_items), total=total_remaining):
            try:
                rec = self.label_single(uid, report)
                results_batch.append(rec)
                all_new_results.append(rec)
                
                # Сохраняем каждые batch_size элементов или в конце
                if len(results_batch) >= batch_size or i == total_remaining - 1:
                    checkpoint.save_batch(results_batch, mode='append' if has_checkpoint or i > 0 else 'overwrite')
                    results_batch = []  # Очищаем батч после сохранения
                    
            except Exception as e:
                print(f"\n✗ Error processing UID {uid}: {e}")
                # Сохраняем ошибку как запись с -1.0 для всех полей
                error_rec = {
                    "StudyInstanceUID": str(uid),
                    **{k: -1.0 for k in KEYS},
                    "processing_time_sec": 0.0,
                    "error": str(e)
                }
                all_new_results.append(error_rec)
                # Можно добавить retry логику здесь если нужно
        
        # Финальное объединение и сохранение
        checkpoint.merge_and_save_final(all_new_results)
        
        # Возвращаем DataFrame с новыми результатами
        if all_new_results:
            return pd.DataFrame(all_new_results)
        return pd.DataFrame()


def main():
    # --- Настройки ---
    INPUT_CSV = "data/train.csv" 
    OUTPUT_CSV = "data/llm_labels_local_api_v1.csv"
    BATCH_SIZE = 10  # Сохранять каждые 10 элементов
    
    df_input = pd.read_csv(INPUT_CSV)
    
    # Определяем колонку с текстом
    text_col = 'Report'
    if text_col not in df_input.columns:
        raise ValueError(f"Column '{text_col}' not found in input CSV. Available: {df_input.columns.tolist()}")
        
    items = list(zip(df_input['StudyInstanceUID'], df_input[text_col]))
    
    # Инициализация
    labeler = LocalApiLabeler(api_url=API_URL)
    
    # Запуск с чекпоинтингом
    df_results = labeler.label_batch(items, OUTPUT_CSV, batch_size=BATCH_SIZE)
    
    if not df_results.empty:
        print("\nFirst 3 new results:")
        print(df_results.head(3))
    
    # Показываем итоговую статистику
    if Path(OUTPUT_CSV).exists():
        df_final = pd.read_csv(OUTPUT_CSV)
        print(f"\n✓ Total records in final CSV: {len(df_final)}")
        print(f"  Columns: {list(df_final.columns)}")

if __name__ == "__main__":
    main()