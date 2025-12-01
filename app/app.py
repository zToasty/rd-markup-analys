import json
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
    st.markdown("Загрузите изображение с дорожной разметкой для анализа через YOLO и VLM.")


@st.cache_resource
def load_models() -> Tuple[YOLOProcessor, VLMProcessor]:
    """Инициализирует и кэширует модели YOLO и VLM."""
    yolo = YOLOProcessor()
    vlm = VLMProcessor()
    yolo.load_model()
    vlm.load_model()
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
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None

    candidate = cleaned[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None

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
) -> Tuple[List[Dict[str, Any]], List[AnalysisResult]]:
    """Выполняет полный цикл: детекция, кроп, анализ VLM."""
    detections = yolo_processor.detect(image, conf_threshold=conf_threshold)
    if not detections:
        return [], []

    cropped_data = yolo_processor.crop_detections(image, detections)
    results: List[AnalysisResult] = []

    for crop in cropped_data:
        vlm_response = vlm_processor.analyze_image(crop["image"], prompt=prompt)
        parsed = parse_vlm_output(vlm_response)
        results.append(
            AnalysisResult(
                detection=crop,
                raw_vlm=vlm_response,
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
        with st.expander(
            f"Объект {idx}: {det['class_name']} (уверенность: {det['confidence']:.2%})",
            expanded=True,
        ):
            col_img, col_text = st.columns([1, 2])

            with col_img:
                st.image(
                    det["image"],
                    caption=f"Обрезанное изображение #{idx}",
                    use_container_width=True,
                )

            with col_text:
                if result.parsed_vlm:
                    data = result.parsed_vlm
                    st.markdown("**Структурированный ответ:**")
                    st.write(f"Износ: {data['wear_percent']}%")
                    st.write(f"Рост риска: {data['accident_risk_increase_percent']}%")
                    st.write(f"Рекомендуемый осмотр через: {data['recommended_repair_days']} дней")
                    st.write(f"Комментарий: {data['comment']}")

                    with st.expander("JSON-ответ"):
                        st.code(
                            json.dumps(data, ensure_ascii=False, indent=2),
                            language="json",
                        )
                else:
                    st.warning("Ответ VLM не соответствует требуемому формату JSON.")
                    st.code(result.raw_vlm)

                st.markdown("**Координаты:**")
                bbox = det["bbox"]
                st.code(f"x1={bbox[0]}, y1={bbox[1]}, x2={bbox[2]}, y2={bbox[3]}")


def render_sidebar() -> None:
    """Отрисовывает вспомогательную информацию в сайдбаре."""
    with st.sidebar:
        st.markdown("### О приложении")
        st.markdown(
            "- YOLO выделяет области разметки на изображении.\n"
            "- VLM выдает структурированное заключение по каждой области."
        )
        st.markdown("### Примечания")
        st.markdown(
            "- Первая загрузка моделей может занять время.\n"
            "- Для корректной работы VLM желательно GPU с 16 ГБ VRAM.\n"
            "- Перед запуском можно отредактировать промпт."
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
                status.text("Детекция объектов (YOLO)...")
                progress.progress(25)
                detections, results = run_analysis(
                    image,
                    conf_threshold,
                    vlm_prompt,
                    yolo_processor,
                    vlm_processor,
                )

                if not detections:
                    progress.progress(100)
                    st.warning("YOLO не обнаружил объектов на изображении.")
                    return

                progress.progress(75)
                status.text("Подготовка отчета...")

                display_detections(detections)
                display_results(results)

                progress.progress(100)
                status.text("Анализ завершен.")
            except Exception as exc:
                st.error(f"Ошибка при анализе: {exc}")
                import traceback

                st.code(traceback.format_exc())
            finally:
                progress.empty()
                status.empty()


if __name__ == "__main__":
    main()
