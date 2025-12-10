import json
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image

from yolo_processor import YOLOProcessor
from vlm_processor import VLMProcessor, DEFAULT_VLM_PROMPT


@dataclass
class AnalysisResult:
    """Результат анализа одной вырезанной области."""

    detection: Dict[str, Any]
    raw_vlm: str
    parsed_vlm: Optional[Dict[str, Any]]


def configure_page() -> None:
    """Базовая настройка Streamlit страницы."""
    st.set_page_config(page_title="Анализ дорожной разметки", layout="wide")
    st.title("Анализ дорожной разметки")
    st.markdown("Загрузите изображение с дорожной разметкой для анализа через YOLO и VLM (Qwen2.5).")


@st.cache_resource
def load_models() -> Tuple[YOLOProcessor, VLMProcessor]:
    """Инициализирует и кэширует модели YOLO и VLM."""
    yolo = YOLOProcessor()
    vlm = VLMProcessor()
    yolo.load_model()
    # vlm.load_model() вызывается автоматически при первом использовании,
    # но можно вызвать здесь, чтобы прогреть сразу:
    # vlm.load_model() 
    return yolo, vlm


def render_settings(default_prompt: str) -> Tuple[float, str]:
    """Отрисовывает настройки и возвращает выбранные параметры."""
    with st.expander("Настройки"):
        conf_threshold = st.slider(
            "Порог уверенности YOLO",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
        )
        vlm_prompt = st.text_area(
            "Промпт для VLM",
            value=default_prompt,
            height=220,
        )
    return conf_threshold, vlm_prompt


def parse_vlm_output(text: str) -> Optional[Dict[str, Any]]:
    """Пытается извлечь единственный JSON-объект из ответа модели."""
    cleaned = text.strip()
    
    # Попытка найти JSON структуру в тексте (от первой { до последней })
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    
    if start == -1 or end == -1 or end < start:
        return None

    candidate = cleaned[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    # Проверка на наличие базовых полей (можно ослабить при необходимости)
    required = {
        "wear_percent",
        "accident_risk_increase_percent",
        "recommended_repair_days",
        "comment",
    }
    
    if not required.issubset(data.keys()):
        return None

    return data


def run_analysis(
    image: Image.Image,
    conf_threshold: float,
    prompt: str,
    yolo_processor: YOLOProcessor,
    vlm_processor: VLMProcessor,
    batch_size: int = 4  # Размер пачки изображений
) -> Tuple[List[Dict[str, Any]], List[AnalysisResult]]:
    """
    Выполняет полный цикл: детекция, кроп, пакетный анализ VLM.
    """
    # 1. Детекция объектов (YOLO)
    detections = yolo_processor.detect(image, conf_threshold=conf_threshold)
    if not detections:
        return [], []

    # 2. Нарезка изображений
    cropped_data = yolo_processor.crop_detections(image, detections)
    
    # Собираем список картинок для отправки в модель
    crop_images_list = [item["image"] for item in cropped_data]
    
    all_vlm_responses = []

    # 3. Пакетная обработка (Batch Processing)
    # Разбиваем список картинок на кусочки по batch_size, чтобы не переполнить память
    for i in range(0, len(crop_images_list), batch_size):
        batch = crop_images_list[i : i + batch_size]
        
        # Вызываем метод analyze_batch из vlm_processor (Qwen)
        batch_responses = vlm_processor.analyze_batch(batch, prompt=prompt)
        all_vlm_responses.extend(batch_responses)

    # 4. Сборка результатов
    results: List[AnalysisResult] = []

    for crop_info, vlm_text in zip(cropped_data, all_vlm_responses):
        parsed = parse_vlm_output(vlm_text)
        results.append(
            AnalysisResult(
                detection=crop_info,
                raw_vlm=vlm_text,
                parsed_vlm=parsed,
            )
        )

    return detections, results


def display_detections(detections: List[Dict[str, Any]]) -> None:
    """Выводит краткую сводку по детекциям YOLO."""
    st.markdown("### Обнаруженные объекты")
    for idx, det in enumerate(detections, start=1):
        st.write(f"{idx}. {det['class_name']} (уверенность: {det['confidence']:.2%})")


def display_results(results: List[AnalysisResult]) -> None:
    """Отрисовывает подробные результаты анализа VLM."""
    if not results:
        return

    st.markdown("---")
    st.markdown("### Результаты анализа")

    for idx, result in enumerate(results, start=1):
        det = result.detection
        
        # Формируем заголовок с кратким итогом
        header_text = f"Объект {idx}: {det['class_name']} (YOLO: {det['confidence']:.2%})"
        if result.parsed_vlm:
            wear = result.parsed_vlm.get('wear_percent', '?')
            header_text += f" | Износ: {wear}%"

        with st.expander(header_text, expanded=True):
            col_img, col_text = st.columns([1, 2])

            with col_img:
                st.image(
                    det["image"],
                    caption=f"Кроп #{idx}",
                    use_container_width=True,
                )

            with col_text:
                if result.parsed_vlm:
                    data = result.parsed_vlm
                    st.markdown("**Структурированный ответ:**")
                    
                    # Цветовое выделение риска
                    risk = data.get('accident_risk_increase_percent', 0)
                    risk_color = "red" if isinstance(risk, (int, float)) and risk > 50 else \
                                 "orange" if isinstance(risk, (int, float)) and risk > 20 else "green"
                    
                    st.write(f"📉 **Износ:** {data.get('wear_percent')}%")
                    st.markdown(f"⚠️ **Рост риска аварии:** :{risk_color}[{risk}%]")
                    st.write(f"🛠 **Рекомендуемый ремонт через:** {data.get('recommended_repair_days')} дней")
                    st.info(f"💬 {data.get('comment')}")

                    with st.popover("Показать JSON"):
                        st.code(
                            json.dumps(data, ensure_ascii=False, indent=2),
                            language="json",
                        )
                else:
                    st.warning("Ответ VLM не соответствует формату JSON.")
                    st.text("Сырой ответ модели:")
                    st.code(result.raw_vlm)

                st.markdown("**Координаты:**")
                bbox = det["bbox"]
                st.caption(f"x1={bbox[0]}, y1={bbox[1]}, x2={bbox[2]}, y2={bbox[3]}")


def render_sidebar() -> None:
    """Отрисовывает вспомогательную информацию и статус VRAM в сайдбаре."""
    with st.sidebar:
        st.markdown("### 🖥️ Мониторинг ресурсов")
        
        if torch.cuda.is_available():
            try:
                device = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device)
                total_memory = props.total_memory
                reserved_memory = torch.cuda.memory_reserved(device)
                allocated_memory = torch.cuda.memory_allocated(device)
                
                # Переводим в ГБ
                total_gb = total_memory / (1024 ** 3)
                reserved_gb = reserved_memory / (1024 ** 3)
                allocated_gb = allocated_memory / (1024 ** 3)
                
                # Вычисляем процент от общей памяти
                usage_percent = reserved_memory / total_memory
                
                st.write(f"**GPU:** {props.name}")
                
                # Метрика
                st.metric(
                    label="Занято VRAM (Reserved)",
                    value=f"{reserved_gb:.2f} GB",
                    delta=f"из {total_gb:.1f} GB",
                    delta_color="off"
                )
                
                # Полоса прогресса
                st.progress(usage_percent)
                
                st.caption(f"Allocated (Tensor data): {allocated_gb:.2f} GB")
                
                # Кнопка очистки
                if st.button("🧹 Очистить кэш CUDA"):
                    torch.cuda.empty_cache()
                    st.toast("Кэш CUDA очищен!")
                    # st.rerun() # Можно раскомментировать для мгновенного обновления UI
            except Exception as e:
                st.error(f"Ошибка мониторинга GPU: {e}")
        else:
            st.warning("⚠️ CUDA не найдена. Работает на CPU.")

        st.markdown("---")
        st.markdown("### О приложении")
        st.markdown(
            "- **YOLO** находит разметку.\n"
            "- **VLM (Qwen2.5)** делает анализ."
        )
        st.markdown("### Инфо")
        st.markdown(
            "- Обработка идет пачками (batch=4).\n"
            "- Требования VRAM: ~6-8 ГБ."
        )


def main() -> None:
    configure_page()
    render_sidebar()

    with st.spinner("Загрузка моделей..."):
        yolo_processor, vlm_processor = load_models()

    uploaded_file = st.file_uploader(
        "Выберите изображение",
        type=["png", "jpg", "jpeg"],
        help="Загрузите изображение с дорожной разметкой",
    )

    if not uploaded_file:
        st.info("Загрузите изображение, чтобы начать анализ.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    col_image, col_controls = st.columns(2)

    with col_image:
        st.subheader("Исходное изображение")
        st.image(image, use_container_width=True)

    with col_controls:
        st.subheader("Управление анализом")
        conf_threshold, vlm_prompt = render_settings(DEFAULT_VLM_PROMPT)

        if st.button("Начать анализ", type="primary"):
            progress = st.progress(0)
            status = st.empty()
            try:
                status.text("🔍 1/3 Детекция объектов (YOLO)...")
                progress.progress(20)
                
                # Запуск анализа (batch_size можно регулировать под память GPU)
                detections, results = run_analysis(
                    image,
                    conf_threshold,
                    vlm_prompt,
                    yolo_processor,
                    vlm_processor,
                    batch_size=4 
                )

                if not detections:
                    progress.progress(100)
                    st.warning("YOLO не обнаружил объектов на изображении.")
                    return

                progress.progress(80)
                status.text("📊 3/3 Формирование отчета...")

                display_detections(detections)
                display_results(results)

                progress.progress(100)
                status.success("Анализ завершен!")
            except Exception as exc:
                st.error(f"Ошибка при анализе: {exc}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                pass


if __name__ == "__main__":
    main()