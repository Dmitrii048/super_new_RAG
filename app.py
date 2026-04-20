import streamlit as st
import os
import json
import sqlite3
import logging
import re
import urllib.parse
import uuid
import requests
from datetime import datetime

# Импорты RAG
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# === 1. НАСТРОЙКИ И СЕКРЕТЫ ===
# Ключ Яндекса берем из секретов Стримлита, если запускаем локально - из переменных
try:
    YANDEX_API_KEY = st.secrets["YANDEX_API_KEY"]
except:
    YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")

FOLDER_ID = os.getenv("FOLDER_ID", "b1g4b3ft2i3eiql7k3p4")
DB_PATH = os.getenv("DB_PATH", "sretensk_db")
TEMPLATES_PATH = os.getenv("TEMPLATES_PATH", "docs/templates")
SITE_INDEX_FILE = "docs/site_index.json"
DB_FILE = "chat_history.db"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === 2. НАСТРОЙКА ИНТЕРФЕЙСА СТРИМЛИТ (СТИЛЬ СДА) ===
st.set_page_config(page_title="Юридический ассистент СДА", page_icon="🎓", layout="centered")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp { background-color: #f9f9f4; }

    .sda-header {
        background: url('https://sdamp.ru/bitrix/templates/main/img/header/day_spring.png') center/cover no-repeat;
        padding: 40px 20px;
        text-align: center;
        border-bottom: 8px solid #942927; 
        margin-top: -80px;
        margin-bottom: 30px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    .sda-header-container {
        display: flex;
        align-items: center;
        justify-content: center;
        max-width: 900px;
        margin: 0 auto;
        gap: 30px;
        background: rgba(255, 255, 255, 0.85); 
        padding: 20px 40px;
        border-radius: 15px;
    }

    .sda-logo { width: 140px; }

    .sda-title {
        color: #942927; 
        font-family: "Times New Roman", Times, serif;
        font-size: 28px; 
        font-weight: bold; 
        text-transform: uppercase;
        line-height: 1.2; 
        margin: 0; 
        text-align: left;
    }

    .sda-subtitle {
        color: #333; 
        font-size: 18px; 
        margin-top: 10px;
        font-family: Arial, sans-serif; 
        text-align: left;
        border-top: 1px solid #ccc; 
        padding-top: 10px;
    }

    .stChatMessage { border-radius: 10px; border: 1px solid #e0e0e0; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="sda-header">
        <div class="sda-header-container">
            <div>
                <h1 class="sda-title">Московская Сретенская<br>Духовная Академия</h1>
                <div class="sda-subtitle">Интеллектуальный юридический ассистент</div>
            </div>
            <img class="sda-logo" src="https://sdamp.ru/bitrix/templates/main/img/logo.png" alt="Логотип СДА">
        </div>
    </div>
""", unsafe_allow_html=True)

# Уникальный ID сессии пользователя для сохранения в БД
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())


# === 3. БАЗА ДАННЫХ SQLITE (СОХРАНЕНО ИЗ ОРИГИНАЛА) ===
def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            username TEXT,
            question TEXT,
            answer TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            question TEXT,
            is_positive INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("✅ База данных чата готова")


init_db()


def save_message(user_id, username, question, answer):
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute(
            'INSERT INTO messages (user_id, username, question, answer) VALUES (?, ?, ?, ?)',
            (user_id, username, question, answer)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Ошибка сохранения: {e}")


def save_feedback(user_id, question, is_positive):
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute(
            'INSERT INTO feedback (user_id, question, is_positive) VALUES (?, ?, ?)',
            (user_id, question, 1 if is_positive else 0)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Ошибка сохранения обратной связи: {e}")


# === 4. ЗАГРУЗКА ИНДЕКСА И ВЕКТОРНОЙ БАЗЫ (СОХРАНЕНО) ===
@st.cache_resource
def load_resources():
    site_index = {'pages': [], 'documents': []}
    if os.path.exists(SITE_INDEX_FILE):
        try:
            with open(SITE_INDEX_FILE, 'r', encoding='utf-8') as f:
                site_index = json.load(f)
            logger.info(
                f"✅ Индекс сайта загружен: {len(site_index['pages'])} страниц, {len(site_index['documents'])} документов")
        except Exception as e:
            logger.error(f"⚠️ Не удалось загрузить индекс сайта: {e}")

    logger.info("⏳ Загружаю базу знаний...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    if os.path.exists(DB_PATH):
        db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        st.error(f"⚠️ Папка с базой данных '{DB_PATH}' не найдена! Поиск работать не будет.")
        return None, site_index

    templates_db_path = DB_PATH + "_templates"
    if os.path.exists(templates_db_path):
        db_templates = FAISS.load_local(templates_db_path, embeddings, allow_dangerous_deserialization=True)
        db.merge_from(db_templates)
        logger.info("✅ База шаблонов загружена")

    return db, site_index


db, site_index = load_resources()


# === 5. ВСЯ ЛОГИКА ПОИСКА (СОХРАНЕНО И УЛУЧШЕНО) ===
def clean_document_name(filename: str) -> str:
    """Улучшение: очистка названий документов для вывода пользователю"""
    name = urllib.parse.unquote(filename)
    name = re.sub(r'\.(docx?|pdf|txt)$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'([а-яёa-z])([А-ЯЁA-Z])', r'\1 \2', name)
    name = re.sub(r'^[\d\s]*СДА\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'^[\d\.\-\s]+', '', name)
    name = re.sub(r'(?<!от\s)(\d{2})[-_.](\d{2})[-_.](\d{4})', r'от \1.\2.\3', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name if name else filename


def find_link_in_index(query: str) -> list:
    query_lower = query.lower()
    results = []
    for page in site_index.get('pages', []):
        title = page.get('title', '').lower()
        url = page.get('url', '').lower()
        if query_lower in title or query_lower in url:
            results.append({'title': page.get('title', 'Страница'), 'url': page.get('url', ''), 'type': 'page'})

    for doc in site_index.get('documents', []):
        name = doc.get('name', '').lower()
        if query_lower in name:
            results.append({'title': doc.get('name', 'Документ'), 'url': doc.get('url', ''),
                            'type': doc.get('type', 'DOC').lower()})
    return results[:5]


def extract_keywords(query: str) -> list:
    stop_words = {'как', 'что', 'где', 'когда', 'почему', 'можно', 'нужно', 'могу', 'ли', 'или', 'и', 'в', 'на', 'по',
                  'для', 'при', 'о', 'об'}
    words = re.findall(r'\b[а-яёА-ЯЁ]{4,}\b', query.lower())
    return [w for w in words if w not in stop_words]


def extract_document_references(docs: list) -> list:
    references = []
    patterns = [r'[Пп]оложение[а-яё\s]*["«]([^"]+)["»]', r'[Пп]риказ[а-яё\s]*№?\s*\d+.*["«]([^"]+)["»]']
    for doc in docs:
        for pattern in patterns:
            references.extend(re.findall(pattern, doc['content']))
    return list(set(references))[:10]


def iterative_search(query: str):
    if not db:
        return [], set()

    found_docs = []
    sources_set = set()

    docs_stage1 = db.similarity_search(query, k=12)
    for d in docs_stage1:
        raw_source = os.path.basename(d.metadata.get('source', 'Неизвестный'))
        source = clean_document_name(raw_source)
        sources_set.add(source)
        found_docs.append({'source': source, 'content': d.page_content, 'stage': 1})

    for term in extract_keywords(query)[:3]:
        docs_stage2 = db.similarity_search(term, k=6)
        for d in docs_stage2:
            raw_source = os.path.basename(d.metadata.get('source', 'Неизвестный'))
            source = clean_document_name(raw_source)
            if source not in [doc['source'] for doc in found_docs]:
                sources_set.add(source)
                found_docs.append({'source': source, 'content': d.page_content, 'stage': 2})

    for doc_ref in extract_document_references(found_docs)[:5]:
        docs_stage3 = db.similarity_search(doc_ref, k=4)
        for d in docs_stage3:
            raw_source = os.path.basename(d.metadata.get('source', 'Неизвестный'))
            source = clean_document_name(raw_source)
            if source not in [doc['source'] for doc in found_docs]:
                sources_set.add(source)
                found_docs.append({'source': source, 'content': d.page_content, 'stage': 3})

    return found_docs, sources_set


def parse_suggestions(answer: str) -> list:
    suggestions = []
    patterns = [
        r'🎯\s*УТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n?(.+)',
        r'УТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n?(.+)',
        r'💡\s*УТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n?(.+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, answer, re.IGNORECASE | re.DOTALL)
        if match:
            suggestions_text = match.group(1).strip()
            questions = re.findall(r'\[([^\]]+)\]|\b([А-Яа-яёЁ].*?\?)', suggestions_text)
            for q in questions:
                if isinstance(q, tuple):
                    for part in q:
                        if part.strip(): suggestions.append(part.strip())
                elif q.strip():
                    suggestions.append(q.strip())
            break

    if not suggestions:
        lines = answer.split('\n')
        for line in reversed(lines[-5:]):
            if '?' in line and len(line) < 100:
                question = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
                if question and question not in suggestions:
                    suggestions.append(question)
    return suggestions[:3]


def clean_answer(answer: str) -> str:
    patterns = [
        r'\n🎯\s*УТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n?.+',
        r'\nУТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n?.+',
        r'\n💡\s*УТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n?.+'
    ]
    for pattern in patterns:
        answer = re.sub(pattern, '', answer, flags=re.DOTALL)
    return answer.strip()


def find_template(user_query: str) -> str | None:
    if not os.path.exists(TEMPLATES_PATH):
        return None
    templates = os.listdir(TEMPLATES_PATH)
    query_lower = user_query.lower()
    keywords_map = {
        'академ': ['академ', 'академическ'],
        'отчисл': ['отчисл', 'выбыт', 'уход'],
        'пересдач': ['пересдач', 'оценк'],
        'дистан': ['дистан', 'онлайн'],
        'справк': ['справк', 'архив'],
        'общежити': ['общежити', 'жиль'],
    }
    for _, search_terms in keywords_map.items():
        if any(term in query_lower for term in search_terms):
            for t in templates:
                if any(term in t.lower() for term in search_terms):
                    return os.path.join(TEMPLATES_PATH, t)
    return None


# === 6. СИСТЕМНЫЙ ПРОМПТ И ВЫЗОВ YANDEX GPT ===
SYSTEM_PROMPT = """
Ты — Интеллектуальный юридический ассистент Сретенской духовной академии (СДА). 
Ты опытный методист с юридическим образованием, работающий в духовном учебном заведении.

КОНТЕКСТ:
- Ты помогаешь студентам, аспирантам и сотрудникам академии
- Твоя задача — давать точные юридические консультации на основе нормативных актов
- Ты должен быть точным, ссылаться на конкретные пункты документов
- Если информации недостаточно — честно об этом скажи

ПРАВИЛА ОТВЕТА ( СТРОГО ):

1. СТРУКТУРА ОТВЕТА ЮРИСТА:

📌 **ЗАКЛЮЧЕНИЕ** (1-2 предложения)[Прямой ответ: ДА/НЕТ/ТРЕБУЕТСЯ/В ЗАВИСИМОСТИ ОТ...]

📖 **ПРАВОВОЕ ОБОСНОВАНИЕ**
   [Развернутый анализ со ссылками на конкретные пункты]
   Пример: "Согласно п. 3.2 Положения о порядке предоставления академических отпусков..."

📋 **ПОРЯДОК ДЕЙСТВИЙ** (если применимо)
   1. [Первый шаг]
   2. [Второй шаг]
   3. [Срок/условия]

📎 **ДОКУМЕНТЫ**
   • [Название документа, номер, пункт]

🔗 **ССЫЛКИ НА САЙТ**
   •[Проверенная ссылка на страницу сайта sdamp.ru, если релевантно]

2. ГЛУБОКИЙ ПОИСК:
   - Анализируй ВСЕ найденные фрагменты
   - Если документ ссылается на другой — найди его

3. ПРОВЕРКА ССЫЛОК (ОБЯЗАТЕЛЬНО!):
   - НЕ выдумывай ссылки — используй ТОЛЬКО те, что найдены в контексте или в индексе сайта
   - Если ссылка не найдена в документах — НЕ включай её в ответ
   - Лучше сказать "ссылку уточните на сайте", чем дать нерабочую ссылку

4. ВАЖНО:
   - Используй ТОЛЬКО информацию из документов
   - Не давай общих советов без ссылок на нормативку
   - НЕ галлюцинируй — не придумывай ссылки и документы

5. УТОЧНЯЮЩИЕ ВОПРОСЫ (ОБЯЗАТЕЛЬНО!):
   В КОНЦЕ каждого ответа добавь 2-3 уточняющих вопроса, которые могут заинтересовать пользователя.
   Формат вывода строго:

   🎯 УТОЧНЯЮЩИЕ ВОПРОСЫ:
   [Вопрос 1?] [Вопрос 2?][Вопрос 3?]

Стиль: Профессиональный юридический, но доброжелательный.
"""


def call_yandex_gpt(messages):
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {
        "Authorization": f"Api-Key {YANDEX_API_KEY}",
        "x-folder-id": FOLDER_ID
    }

    yandex_messages = []
    for role, text in messages:
        yandex_role = "system" if role == "system" else "user"
        yandex_messages.append({"role": yandex_role, "text": text})

    payload = {
        "modelUri": f"gpt://{FOLDER_ID}/yandexgpt/latest",
        "completionOptions": {"stream": False, "temperature": 0.2, "maxTokens": "2000"},
        "messages": yandex_messages
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()['result']['alternatives'][0]['message']['text']
    else:
        raise Exception(f"Ошибка YandexGPT: {response.text}")


def get_rag_response(question: str):
    docs, sources = iterative_search(question)

    site_links = find_link_in_index(question)
    site_context = ""
    if site_links:
        site_context = "\n📎 РЕЛЕВАНТНЫЕ ССЫЛКИ НА САЙТЕ (ПРОВЕРЕННЫЕ):\n"
        for link in site_links:
            site_context += f"- {link['title']}: {link['url']}\n"

    if not docs:
        if site_links:
            links_text = "\n".join([f"🔗 [{link['title']}]({link['url']})" for link in site_links])
            return f"В базе документов не найдено, но есть информация на сайте:\n{links_text}", [], ""
        return "😔 В базе знаний не найдено релевантных документов.", [], ""

    docs.sort(key=lambda x: x['stage'])
    context = "\n\n".join(
        [f"--- ФРАГМЕНТ {i + 1} ({d['source']}) ---\n{d['content']}" for i, d in enumerate(docs[:15])])

    if site_context:
        context += site_context

    messages = [
        ("system", SYSTEM_PROMPT),
        ("human",
         f"КОНТЕКСТ:\n{context}\n\nВОПРОС: {question}\n\nДай структурированный ответ с уточняющими вопросами в конце. Проверь, что все ссылки реальны!")
    ]

    try:
        raw_answer = call_yandex_gpt(messages)

        sources_text = "\n".join([f"• {s}" for s in sources])
        if site_links:
            links_text = "\n".join([f"🔗 [{link['title']}]({link['url']})" for link in site_links])
            sources_text += f"\n\n🌐 *Проверенные ссылки на сайт:*\n{links_text}"

        suggestions = parse_suggestions(raw_answer)
        answer = clean_answer(raw_answer)

        return answer, suggestions, sources_text
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        return f"Произошла ошибка при обращении к ИИ: {e}", [], ""


# === 7. ГЛАВНЫЙ ИНТЕРФЕЙС И ЛОГИКА ЧАТА ===

# Инициализация истории сообщений
if "messages" not in st.session_state:
    welcome_text = (
        "👋 Здравствуйте! Я — Интеллектуальный помощник Сретенской духовной академии.\n"
        "Я знаю всё о Положениях, Приказах и Уставе.\n\n"
        "📖 **Чем я могу помочь?**\n"
        "• Отвечу на вопросы о правилах\n"
        "• Помогу найти нужный документ\n"
        "• Подберу шаблон заявления"
    )
    st.session_state.messages = [
        {"role": "assistant", "content": welcome_text, "sources": None, "template": None,
         "suggestions": ["Как оформить академический отпуск?", "Какие документы нужны для отчисления?",
                         "Как получить справку об обучении?"]}
    ]

# Отрисовка всех сообщений из истории
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Кнопка скачивания шаблона (если был найден)
        if msg.get("template") and os.path.exists(msg["template"]):
            with open(msg["template"], "rb") as f:
                st.download_button("📥 Скачать документ", f, file_name=os.path.basename(msg["template"]), key=f"dl_{i}")

        # Источники (если есть)
        if msg.get("sources"):
            with st.expander("📚 Использованные документы"):
                st.markdown(msg["sources"])

        # Кнопки подсказок (только у последнего сообщения ассистента)
        if msg.get("suggestions") and msg["role"] == "assistant" and i == len(st.session_state.messages) - 1:
            st.markdown("💡 *Возможно, вас также заинтересует:*")
            for sug in msg["suggestions"]:
                if st.button(sug, key=f"sug_{i}_{sug}"):
                    st.session_state.suggestion_clicked = sug
                    st.rerun()

# Ввод нового вопроса
prompt = st.chat_input("Напишите ваш вопрос (например, 'Как оформить академ?')...")

# Обработка нажатия на кнопку-подсказку
if "suggestion_clicked" in st.session_state:
    prompt = st.session_state.suggestion_clicked
    del st.session_state.suggestion_clicked

if prompt:
    # 1. Показываем вопрос пользователя
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Генерируем ответ ассистента
    with st.chat_message("assistant"):
        with st.spinner("Изучаю нормативные акты СДА..."):

            # Проверка на запрос шаблона
            petition_keywords = ['прошение', 'заявление', 'бланк', 'шаблон', 'образец']
            need_template = any(kw in prompt.lower() for kw in petition_keywords)
            template_path = find_template(prompt) if need_template else None

            if template_path:
                # ВЕТКА ШАБЛОНОВ
                filename = os.path.basename(template_path)
                ans_text = f"📄 **Нашёл подходящий шаблон:** {filename}\nЗаполните его и передайте в учебную часть."
                st.markdown(ans_text)
                with open(template_path, "rb") as f:
                    st.download_button("📥 Скачать документ", f, file_name=filename, key="dl_new")

                st.session_state.messages.append({"role": "assistant", "content": ans_text, "template": template_path})
                save_message(st.session_state.user_id, "WebUser", prompt, ans_text)
                st.rerun()

            else:
                # ВЕТКА RAG И YANDEX GPT
                answer, suggestions, sources_text = get_rag_response(prompt)

                st.markdown(answer)
                if sources_text:
                    with st.expander("📚 Использованные документы"):
                        st.markdown(sources_text)

                # Обратная связь
                col1, col2, col3 = st.columns([1, 1, 4])
                with col1:
                    if st.button("👍 Полезно", key="up_new"):
                        save_feedback(st.session_state.user_id, prompt, True)
                with col2:
                    if st.button("👎 Нет", key="down_new"):
                        save_feedback(st.session_state.user_id, prompt, False)

                # Сохраняем в сессию
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources_text,
                    "suggestions": suggestions
                })

                # Сохраняем в базу данных
                full_log = f"{answer}\n\nИсточники:\n{sources_text}"
                save_message(st.session_state.user_id, "WebUser", prompt, full_log)

                st.rerun()