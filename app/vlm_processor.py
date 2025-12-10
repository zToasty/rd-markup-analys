import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

DEFAULT_VLM_PROMPT = (
    "Ты — алгоритм компьютерного зрения для оценки износа разметки. Твоя задача — классифицировать состояние по строгой шкале.\n\n"
    "ИСПОЛЬЗУЙ ЭТУ ШКАЛУ (Rubric):\n"
    "1. ОТЛИЧНО (0-15%): Линии яркие, края четкие. Цвет (белый/желтый) насыщенный. (Сюда относится и свежая 'вафельная' разметка).\n"
    "2. НОРМА (15-35%): Разметка видна хорошо, но цвет стал матовым, края слегка размыты. Разрывов нет.\n"
    "3. ИЗНОС (35-60%): Виден асфальт сквозь краску (прозрачность). Края неровные. Есть мелкие проплешины.\n"
    "4. ПЛОХО (60-85%): Значительные куски краски отсутствуют. Разметка трудно различима на фоне дороги.\n"
    "5. КРИТИЧНО (85-100%): Разметки почти не осталось.\n\n"
    "ВАЖНО:\n"
    "- Если это желтая сетка ('вафельница') и линии сплошные — это 0-10% износа, это НЕ трещины.\n"
    "- Если краска 'рябит' или просвечивает — ставь минимум 40%.\n\n"
    "Верни JSON:\n"
    "{\n"
    '  "wear_percent": int,\n'
    '  "accident_risk_increase_percent": int,\n'
    '  "recommended_repair_days": int,\n'
    '  "comment": "строгий вывод на русском, опираясь на шкалу"\n'
    "}"
)
class VLMProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Если будет падать по памяти, замените на "Qwen/Qwen2-VL-2B-Instruct"
        self.model_id = "Qwen/Qwen2.5-VL-3B-Instruct" 
        self.processor = None
        self.model = None
        self._loaded = False
    
    def load_model(self):
        """Загружает модель VLM"""
        if self._loaded:
            return
        
        print(f"🔄 Загрузка модели VLM: {self.model_id}...")
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                attn_implementation="flash_attention_2" if self.device == "cuda" else None
            )
        except Exception:
            # Фолбек, если нет flash_attention
            print("⚠️ Flash Attention не найден, загрузка в обычном режиме...")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype="auto",
                device_map="auto"
            )
            
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self._loaded = True
        print("✅ Модель VLM готова")

    def analyze_image(self, image: Image.Image, prompt: str = DEFAULT_VLM_PROMPT) -> str:
        """Обработка одной картинки (базовая функция)."""
        if not self._loaded:
            self.load_model()
            
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Подготовка через qwen utils
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to(self.model.device)

        # Генерация с небольшим запасом токенов
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=256,
                do_sample=False  # Отключаем случайность для строгости JSON
            )
        
        # Правильное декодирование (обрезаем вход)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return self._clean_json(output_text)

    def analyze_batch(self, crops: list, prompt: str = DEFAULT_VLM_PROMPT) -> list:
        """
        БЕЗОПАСНАЯ пакетная обработка.
        Вместо того чтобы пихать всё в один тензор (что ломает Qwen из-за разных размеров),
        мы проходим циклом. Это чуть медленнее, но гарантирует, что не будет пустых ответов.
        """
        results = []
        for crop_data in crops:
            # Извлекаем картинку, если передан словарь, или берем как есть
            img = crop_data['image'] if isinstance(crop_data, dict) else crop_data
            
            try:
                # Вызываем анализ для одной картинки
                res = self.analyze_image(img, prompt)
                results.append(res)
            except Exception as e:
                print(f"❌ Ошибка анализа VLM: {e}")
                results.append("{}") # Возвращаем пустой JSON в случае сбоя
                
        return results

    def _clean_json(self, text: str) -> str:
        """Чистит ответ от markdown-оберток."""
        text = text.strip()
        # Убираем ```json и ```
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
            
        if text.endswith("```"):
            text = text[:-3]
            
        return text.strip()