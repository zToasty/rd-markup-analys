from ultralytics import YOLO
from PIL import Image
import numpy as np
from pathlib import Path

class YOLOProcessor:
    def __init__(self, model_path: str = None):
        if model_path is None:
            yolo_dir = Path(__file__).parent.parent / "yolo"
            model_files = list(yolo_dir.glob("*.pt"))
            if model_files:
                self.model_path = str(model_files[0])
            else:
                self.model_path = "yolov8n.pt"
        else:
            self.model_path = model_path
        
        self.model = None
        self._loaded = False
    
    def load_model(self):
        if self._loaded: return
        print(f"🚀 Загрузка YOLO: {self.model_path}...")
        self.model = YOLO(self.model_path)
        self._loaded = True
        print("✅ YOLO загружена")
    
    def detect(self, image: Image.Image, conf_threshold: float = 0.25) -> list:
        if not self._loaded: self.load_model()
        results = self.model(image, conf=conf_threshold)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(box.conf[0].cpu().numpy()),
                    'class_id': int(box.cls[0].cpu().numpy()),
                    'class_name': self.model.names[int(box.cls[0].cpu().numpy())]
                })
        return detections
    
    def crop_detections(self, image: Image.Image, detections: list, padding: int = 40) -> list:
        """
        Обрезает изображение с отступом (padding), чтобы VLM видела контекст (асфальт).
        """
        w, h = image.size
        cropped_images = []
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Добавляем отступ, но не выходим за границы фото
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            cropped = image.crop((x1, y1, x2, y2))
            
            # Фильтр совсем мелкого мусора (если кроп меньше 50x50 пикселей - пользы от него нет)
            if cropped.size[0] < 50 or cropped.size[1] < 50:
                continue

            cropped_images.append({
                'image': cropped,
                'bbox': det['bbox'], # Сохраняем оригинальные координаты
                'confidence': det['confidence'],
                'class_name': det['class_name']
            })
        
        return cropped_images