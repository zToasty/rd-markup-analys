import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq


class VLMProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
        self.processor = None
        self.model = None
        self._loaded = False
    
    def load_model(self):
        """Загружает модель VLM"""
        if self._loaded:
            return
        
        print(f"🚀 Загрузка модели: {self.model_id}...")
        print(f"🤖 Используемое устройство: {self.device}")
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        self._loaded = True
        print("✅ Модель VLM загружена")
    
    def analyze_image(self, image: Image.Image, prompt: str = "Опиши состояние дорожной разметки на этом изображении. Она новая или старая?") -> str:
        """
        Анализирует изображение с помощью VLM
        
        Args:
            image: PIL Image для анализа
            prompt: Текст промпта для модели
            
        Returns:
            str: Результат анализа
        """
        if not self._loaded:
            self.load_model()
        
        # Форматирование промпта для Llava-NeXT (1.6)
        full_prompt = f"[INST] <image>\n{prompt} [/INST]"
        
        # Обработка изображения
        inputs = self.processor(
            images=image,
            text=full_prompt,
            return_tensors="pt"
        )
        
        # Переносим все данные на устройство
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Генерация ответа
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
        
        # Декодирование и очистка ответа
        answer = self.processor.decode(output[0], skip_special_tokens=True)
        
        # Очистка ответа от промпта
        if "[/INST]" in answer:
            final_answer = answer.split("[/INST]")[-1].strip()
        else:
            final_answer = answer.strip()
        
        return final_answer

