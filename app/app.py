import streamlit as st
from PIL import Image
import io
from yolo_processor import YOLOProcessor
from vlm_processor import VLMProcessor


# Настройка страницы
st.set_page_config(
    page_title="Анализ дорожной разметки",
    page_icon="🛣️",
    layout="wide"
)

st.title("🛣️ Анализ дорожной разметки")
st.markdown("Загрузите изображение для анализа через YOLO и VLM модели")

# Инициализация моделей в session state
@st.cache_resource
def load_models():
    """Загружает модели один раз и кэширует их"""
    # YOLOProcessor автоматически найдет модель в папке yolo/ если она там есть
    yolo = YOLOProcessor()
    vlm = VLMProcessor()
    return yolo, vlm

# Загрузка моделей
with st.spinner("Загрузка моделей..."):
    yolo_processor, vlm_processor = load_models()
    yolo_processor.load_model()
    vlm_processor.load_model()

# Загрузка изображения
uploaded_file = st.file_uploader(
    "Выберите изображение",
    type=['png', 'jpg', 'jpeg'],
    help="Загрузите изображение с дорожной разметкой"
)

if uploaded_file is not None:
    # Отображение загруженного изображения
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Исходное изображение")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("Результаты анализа")
        
        # Настройки
        with st.expander("⚙️ Настройки"):
            conf_threshold = st.slider(
                "Порог уверенности YOLO",
                min_value=0.1,
                max_value=1.0,
                value=0.25,
                step=0.05
            )
            vlm_prompt = st.text_area(
                "Промпт для VLM",
                value="Опиши состояние дорожной разметки на этом изображении. Она новая или старая?",
                height=100
            )
        
        # Кнопка анализа
        if st.button("🔍 Начать анализ", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Шаг 1: Детекция YOLO
                status_text.text("🔍 Выполняется детекция объектов (YOLO)...")
                progress_bar.progress(20)
                
                detections = yolo_processor.detect(image, conf_threshold=conf_threshold)
                
                if not detections:
                    st.warning("⚠️ YOLO не обнаружил объектов на изображении")
                    progress_bar.progress(100)
                else:
                    st.success(f"✅ Найдено объектов: {len(detections)}")
                    
                    # Отображение детекций
                    st.markdown("### Обнаруженные объекты:")
                    for i, det in enumerate(detections, 1):
                        st.write(f"**{i}. {det['class_name']}** (уверенность: {det['confidence']:.2%})")
                    
                    # Шаг 2: Обрезка изображений
                    status_text.text("✂️ Обрезка изображений...")
                    progress_bar.progress(40)
                    
                    cropped_data = yolo_processor.crop_detections(image, detections)
                    
                    # Шаг 3: Анализ через VLM
                    status_text.text("🧠 Анализ через VLM...")
                    progress_bar.progress(60)
                    
                    results = []
                    for idx, crop_data in enumerate(cropped_data, 1):
                        status_text.text(f"🧠 Анализ объекта {idx}/{len(cropped_data)}...")
                        progress_bar.progress(60 + int(30 * idx / len(cropped_data)))
                        
                        cropped_image = crop_data['image']
                        vlm_result = vlm_processor.analyze_image(
                            cropped_image,
                            prompt=vlm_prompt
                        )
                        
                        results.append({
                            'detection': crop_data,
                            'vlm_analysis': vlm_result
                        })
                    
                    # Отображение результатов
                    status_text.text("✅ Анализ завершен!")
                    progress_bar.progress(100)
                    
                    st.markdown("---")
                    st.markdown("### 📊 Результаты анализа:")
                    
                    for idx, result in enumerate(results, 1):
                        with st.expander(
                            f"Объект {idx}: {result['detection']['class_name']} "
                            f"(уверенность: {result['detection']['confidence']:.2%})",
                            expanded=True
                        ):
                            col_img, col_text = st.columns([1, 2])
                            
                            with col_img:
                                st.image(
                                    result['detection']['image'],
                                    caption=f"Обрезанное изображение #{idx}",
                                    use_container_width=True
                                )
                            
                            with col_text:
                                st.markdown("**Анализ VLM:**")
                                st.info(result['vlm_analysis'])
                                
                                st.markdown("**Координаты:**")
                                bbox = result['detection']['bbox']
                                st.code(f"x1={bbox[0]}, y1={bbox[1]}, x2={bbox[2]}, y2={bbox[3]}")
            
            except Exception as e:
                st.error(f"❌ Ошибка при обработке: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
            
            finally:
                progress_bar.empty()
                status_text.empty()

else:
    st.info("👆 Пожалуйста, загрузите изображение для начала анализа")

# Информация о моделях
with st.sidebar:
    st.markdown("### ℹ️ О приложении")
    st.markdown("""
    Это приложение выполняет анализ дорожной разметки в два этапа:
    
    1. **YOLO детекция** - обнаружение объектов на изображении
    2. **VLM анализ** - детальный анализ обрезанных областей
    
    **Используемые модели:**
    - YOLO: YOLOv8n (Ultralytics)
    - VLM: LLaVA v1.6 Mistral 7B
    """)
    
    st.markdown("### 📝 Примечания")
    st.markdown("""
    - Первая загрузка моделей может занять некоторое время
    - Для работы VLM требуется значительный объем памяти
    - Рекомендуется использовать GPU для ускорения обработки
    """)

