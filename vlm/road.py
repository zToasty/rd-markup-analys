import torch
from PIL import Image
# Используем AutoProcessor и АВТОМАТИЧЕСКИЙ класс для VLM (Vision to Sequence)
from transformers import AutoProcessor, AutoModelForVision2Seq # Ключевое изменение здесь!

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "llava-hf/llava-v1.6-mistral-7b-hf" 

print(f"🚀 Загрузка модели: {model_id}...")
print(f"🤖 Используемое устройство: {device}")


processor = AutoProcessor.from_pretrained(model_id)

# !!! КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: AutoModelForVision2Seq !!!
# Этот класс специально разработан для мультимодальных моделей VLM
model = AutoModelForVision2Seq.from_pretrained( 
    model_id,
    dtype=torch.float16 if device=="cuda" else torch.float32, 
    low_cpu_mem_usage=True
).to(device)

# --- Далее код, который вы уже использовали: ---

# Убедитесь, что этот путь верен
try:
    image = Image.open("../image.png").convert("RGB")
except FileNotFoundError:
    print("❌ Ошибка: Файл изображения '../image.png' не найден!")
    exit()

# Форматирование промпта для Llava-NeXT (1.6)
prompt_text = "Опиши состояние дорожной разметки на этом изображении. Она новая или старая?"
full_prompt = f"[INST] <image>\n{prompt_text} [/INST]"

print("🖼️ Изображение загружено. Выполняется анализ...")

# Правильный вызов
inputs = processor(
    images=image,
    text=full_prompt,
    return_tensors="pt"
)

# Переносим все данные на устройство (CPU/GPU)
inputs = {k: v.to(device) for k, v in inputs.items()} 

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False
    )

# Декодирование и очистка ответа
answer = processor.decode(output[0], skip_special_tokens=True)

# Очистка ответа от промпта
if "[/INST]" in answer:
    final_answer = answer.split("[/INST]")[-1].strip()
else:
    final_answer = answer.strip()

print("\n--- 🧠 РЕЗУЛЬТАТ АНАЛИЗА ---")
print(final_answer)
print("-----------------------------\n")