from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
from pathlib import Path


class YOLOProcessor:
    def __init__(self, model_path: str = None):
        """
        Инициализирует процессор YOLO
        
        Args:
            model_path: Путь к модели YOLO или имя предобученной модели.
                       Если None, автоматически ищет модель в папке yolo/
        """
        if model_path is None:
            # Ищем модель в папке yolo/
            yolo_dir = Path(__file__).parent.parent / "yolo"
            model_files = list(yolo_dir.glob("*.pt"))
            
            if model_files:
                # Используем первую найденную модель
                self.model_path = str(model_files[0])
                print(f"📁 Найдена модель в папке yolo/: {self.model_path}")
            else:
                # Используем стандартную модель
                self.model_path = "yolov8n.pt"
                print("📦 Используется стандартная модель YOLO")
        else:
            self.model_path = model_path
        
        self.model = None
        self._loaded = False
    
    def load_model(self):
        """Загружает модель YOLO"""
        if self._loaded:
            return
        
        print(f"🚀 Загрузка модели YOLO: {self.model_path}...")
        self.model = YOLO(self.model_path)
        self._loaded = True
        print("✅ Модель YOLO загружена")
    
    def detect(self, image: Image.Image, conf_threshold: float = 0.25) -> list:
        """
        Выполняет детекцию объектов на изображении
        
        Args:
            image: PIL Image для детекции
            conf_threshold: Порог уверенности для детекции
            
        Returns:
            list: Список детекций, каждая содержит:
                - bbox: (x1, y1, x2, y2) координаты bounding box
                - confidence: Уверенность детекции
                - class_id: ID класса
                - class_name: Имя класса
        """
        if not self._loaded:
            self.load_model()
        
        # Выполняем детекцию
        results = self.model(image, conf=conf_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Получаем координаты
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[class_id]
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        return detections
    
    def crop_detections(self, image: Image.Image, detections: list) -> list:
        """
        Обрезает изображение по найденным детекциям
        
        Args:
            image: Исходное PIL Image
            detections: Список детекций от метода detect()
            
        Returns:
            list: Список обрезанных изображений (PIL Image)
        """
        cropped_images = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            # Обрезаем изображение
            cropped = image.crop((x1, y1, x2, y2))
            cropped_images.append({
                'image': cropped,
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'class_name': det['class_name']
            })
        
        return cropped_images

