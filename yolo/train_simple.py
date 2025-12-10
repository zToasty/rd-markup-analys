#!/usr/bin/env python3
"""
Упрощенная версия: для уже скачанных датасетов
Положи датасеты в папки dataset1/, dataset2/ и т.д.
"""

import os
import shutil
from pathlib import Path
import yaml
from ultralytics import YOLO

# Пути к твоим скачанным датасетам
DATASET_PATHS = [
    "dataset1",  # Путь к первому датасету
    "dataset2",  # Путь ко второму датасету
    # Добавь больше если нужно
]

OUTPUT_DIR = Path("merged_dataset")

def merge_and_relabel(dataset_paths, output_dir):
    """Объединяет датасеты и переводит все в класс 0"""
    output_dir.mkdir(exist_ok=True)

    for split in ['train', 'valid', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    img_counter = 0

    for ds_path in dataset_paths:
        ds_path = Path(ds_path)

        for split in ['train', 'valid', 'test']:
            img_dir = ds_path / split / 'images'
            label_dir = ds_path / split / 'labels'

            if not img_dir.exists():
                continue

            for img_file in img_dir.glob('*.[jp][pn]g'):
                # Копируем изображение
                new_name = f"img_{img_counter:06d}{img_file.suffix}"
                shutil.copy(img_file, output_dir / split / 'images' / new_name)

                # Меняем все классы на 0
                label_file = label_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    new_label = output_dir / split / 'labels' / f"img_{img_counter:06d}.txt"

                    with open(label_file, 'r') as f_in, open(new_label, 'w') as f_out:
                        for line in f_in:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                parts[0] = '0'  # Все классы -> 0
                                f_out.write(' '.join(parts) + '\n')

                img_counter += 1

    print(f"✅ Объединено {img_counter} изображений")
    return img_counter

# Создаем data.yaml
def create_config(output_dir):
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'nc': 1,
        'names': ['road_marking']
    }

    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)

    return yaml_path

# Тренируем
if __name__ == "__main__":
    print("🔄 Объединяю датасеты...")
    merge_and_relabel(DATASET_PATHS, OUTPUT_DIR)

    print("📝 Создаю конфиг...")
    yaml_path = create_config(OUTPUT_DIR)

    print("🚀 Начинаю тренировку...")
    model = YOLO('yolov8n.pt')
    model.train(
        data=str(yaml_path),
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        project='road_marking_training',
        name='exp'
    )

    print("✅ Готово! Модель: road_marking_training/exp/weights/best.pt")
