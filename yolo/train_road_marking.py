import os
import shutil
from pathlib import Path
import yaml

# Установка зависимостей (запусти один раз)
# pip install roboflow ultralytics opencv-python

from roboflow import Roboflow
from ultralytics import YOLO

# ============== КОНФИГУРАЦИЯ ==============
ROBOFLOW_API_KEY = "TXGdHa3eiUrFZuVZblhm"  # Получи на roboflow.com
DATASETS = [
    {
        "workspace": "emds-corecenter-pbjjv",
        "project": "road-markings-detection-ceymo",
        "version": 1
    },
    {
        "workspace": "lr-tdx",
        "project": "road-markings-0n4es",
        "version": 2
    },
    {
        "workspace": "road-mark",
        "project": "road-marking-cnzo6",
        "version": 2
    }
]

OUTPUT_DIR = Path("merged_dataset")
TRAIN_EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640

# ============== ФУНКЦИИ ==============

def download_datasets(api_key, datasets):
    """Скачивает датасеты с Roboflow"""
    rf = Roboflow(api_key=api_key)
    downloaded = []

    for ds in datasets:
        print(f"\n📥 Скачиваю {ds['project']}...")
        try:
            project = rf.workspace(ds['workspace']).project(ds['project'])
            version = project.version(ds['version'])
            dataset = version.download("yolov8", location=f"./temp_{ds['project']}")
            downloaded.append(f"./temp_{ds['project']}")
            print(f"✅ Скачан: {ds['project']}")
        except Exception as e:
            print(f"❌ Ошибка при скачивании {ds['project']}: {e}")

    return downloaded

def merge_datasets(dataset_paths, output_dir):
    """Объединяет датасеты и переводит все классы в один"""
    output_dir.mkdir(exist_ok=True)

    for split in ['train', 'valid', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    img_counter = 0

    for ds_path in dataset_paths:
        ds_path = Path(ds_path)
        print(f"\n📦 Обрабатываю {ds_path.name}...")

        for split in ['train', 'valid', 'test']:
            img_dir = ds_path / split / 'images'
            label_dir = ds_path / split / 'labels'

            if not img_dir.exists():
                continue

            # Копируем изображения
            for img_file in img_dir.glob('*.[jp][pn]g'):
                new_name = f"img_{img_counter:06d}{img_file.suffix}"
                shutil.copy(img_file, output_dir / split / 'images' / new_name)

                # Обрабатываем аннотации - все классы становятся 0
                label_file = label_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    new_label = output_dir / split / 'labels' / f"img_{img_counter:06d}.txt"

                    with open(label_file, 'r') as f_in, open(new_label, 'w') as f_out:
                        for line in f_in:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                # Меняем класс на 0, оставляем координаты
                                parts[0] = '0'
                                f_out.write(' '.join(parts) + '\n')

                img_counter += 1

        print(f"✅ Обработано {img_counter} изображений")

    return img_counter

def create_yaml(output_dir, total_images):
    """Создает data.yaml для YOLO"""
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 1,  # Один класс
        'names': ['road_marking']
    }

    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"\n📄 Создан {yaml_path}")
    print(f"📊 Всего изображений: {total_images}")
    return yaml_path

def train_yolo(data_yaml):
    """Тренирует YOLOv8 на объединенном датасете"""
    print("\n🚀 Начинаю тренировку YOLOv8...")

    # Загружаем предобученную модель
    model = YOLO('yolov8n.pt')  # Можно yolov8s.pt, yolov8m.pt для большей точности

    # Тренируем
    results = model.train(
        data=str(data_yaml),
        epochs=TRAIN_EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=0,  # GPU 0, или 'cpu' для CPU
        patience=20,  # Early stopping
        save=True,
        project='road_marking_training',
        name='exp',
        exist_ok=True
    )

    print("\n✅ Тренировка завершена!")
    print(f"📁 Модель сохранена в: road_marking_training/exp/weights/best.pt")

    return results

def cleanup(dataset_paths):
    """Удаляет временные папки"""
    for ds_path in dataset_paths:
        if os.path.exists(ds_path):
            shutil.rmtree(ds_path)
            print(f"🗑️  Удалена временная папка: {ds_path}")

# ============== ОСНОВНОЙ КОД ==============

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 Автоматическая тренировка YOLO для детекции разметки")
    print("=" * 60)

    # Шаг 1: Скачивание датасетов
    print("\n[1/4] Скачивание датасетов с Roboflow...")
    downloaded_paths = download_datasets(ROBOFLOW_API_KEY, DATASETS)

    if not downloaded_paths:
        print("❌ Не удалось скачать ни один датасет. Проверь API ключ.")
        exit(1)

    # Шаг 2: Объединение датасетов
    print("\n[2/4] Объединение датасетов в один класс...")
    total_imgs = merge_datasets(downloaded_paths, OUTPUT_DIR)

    # Шаг 3: Создание конфига
    print("\n[3/4] Создание конфигурации...")
    yaml_path = create_yaml(OUTPUT_DIR, total_imgs)

    # Шаг 4: Тренировка
    print("\n[4/4] Тренировка модели...")
    train_yolo(yaml_path)

    # Очистка
    print("\n🧹 Очистка временных файлов...")
    cleanup(downloaded_paths)

    print("\n" + "=" * 60)
    print("🎉 ВСЁ ГОТОВО!")
    print("=" * 60)
    print("\nИспользуй обученную модель:")
    print("  from ultralytics import YOLO")
    print("  model = YOLO('road_marking_training/exp/weights/best.pt')")
    print("  results = model('your_image.jpg')")
