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

# === 1. НАСТРОЙКИ И КОНФИГУРАЦИЯ ===
try:
    YANDEX_API_KEY = st.secrets["YANDEX_API_KEY"]
except:
    YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")

# ИСПРАВЛЕННЫЙ ID КАТАЛОГА (из твоей ошибки в логах)
FOLDER_ID = "b1g6jhk9eapudn6lom6c" 

DB_PATH = os.getenv("DB_PATH", "sretensk_db")
TEMPLATES_PATH = os.getenv("TEMPLATES_PATH", "docs/templates")
SITE_INDEX_FILE = "docs/site_index.json"
DB_FILE = "chat_history.db"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === 2. ФИРМЕННЫЙ ИНТЕРФЕЙС СДА (ИСПРАВЛЕННЫЙ ДИЗАЙН) ===
st.set_page_config(page_title="Юридический ассистент СДА", page_icon="🎓", layout="centered")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Убираем пустые места сверху и снизу */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        max-width: 950px;
    }
    
    .stApp { background-color: #FCFCFA; }
    
    /* Дизайн баннера в стиле сайта СДА */
    .sda-banner {
        background: linear-gradient(90deg, rgba(255,255,255,1) 0%, rgba(255,255,255,0.95) 50%, rgba(255,255,255,0) 100%), 
                    url('https://sdamp.ru/bitrix/templates/main/img/header/day_spring.png');
        background-size: cover;
        background-position: right center;
        border-left: 8px solid #942927; 
        border-radius: 12px;
        padding: 30px 40px;
        display: flex;
        align-items: center;
        gap: 25px;
        margin-top: 0px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    }
    
    .sda-logo { width: 100px; height: auto; flex-shrink: 0; }
    .sda-text-block { display: flex; flex-direction: column; }
    
    .sda-title {
        font-family: 'Times New Roman', Times, serif;
        color: #1a2a44; font-size: 26px; font-weight: 900;
        text-transform: uppercase; line-height: 1.1; margin: 0;
        letter-spacing: 0.5px;
    }
    
    .sda-subtitle {
        font-family: 'Arial', sans-serif;
        color: #942927; font-size: 15px; margin-top: 10px;
        font-weight: 600; text-transform: uppercase; letter-spacing: 1px;
    }
    
    /* Стилизация чата */
    .stChatMessage { border-radius: 10px; padding: 15px; border: 1px solid #eaeaea; margin-bottom: 10px; }
    .stButton>button { 
        background-color: #f0f2f6; 
        border-radius: 20px; 
        border: 1px solid #ddd; 
        color: #1a2a44; 
        width: 100%;
        text-align: left;
        padding: 10px 20px;
    }
    .stButton>button:hover { border-color: #c5a059; color: #942927; background-color: #fff; }
    </style>

    <div class="sda-banner">
        <img class="sda-logo" src="https://sdamp.ru/bitrix/templates/main/img/logo.png" alt="Логотип СДА">
        <div class="sda-text-block">
            <div class="sda-title">Московская Сретенская<br>Духовная Академия</div>
            <div class="sda-subtitle">Интеллектуальный Юридический Ассистент</div>
        </div>
    </div>
""", unsafe_allow_html=True)

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

# === 3. БАЗА ДАННЫХ SQLITE (ПОЛНАЯ ЛОГИКА) ===
def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute('''CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, username TEXT, question TEXT, answer TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, question TEXT, is_positive INTEGER, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

def save_message(user_id, question, answer):
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute('INSERT INTO messages (user_id, username, question, answer) VALUES (?, ?, ?, ?)', (user_id, "WebUser", question, answer))
        conn.commit()
        conn.close()
    except Exception as e: logger.error(f"Ошибка сохранения в БД: {e}")

def save_feedback(user_id, question, is_positive):
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute('INSERT INTO feedback (user_id, question, is_positive) VALUES (?, ?, ?)', (user_id, question, 1 if is_positive else 0))
        conn.commit()
        conn.close()
    except Exception as e: logger.error(f"Ошибка сохранения feedback: {e}")

# === 4. ЗАГРУЗКА РЕСУРСОВ ===
@st.cache_resource
def load_resources():
    site_index = {'pages':[], 'documents':[]}
    if os.path.exists(SITE_INDEX_FILE):
        try:
            with open(SITE_INDEX_FILE, 'r', encoding='utf-8') as f:
                site_index = json.load(f)
            logger.info("✅ Индекс сайта успешно загружен")
        except: pass

    logger.info("⏳ Инициализация эмбеддингов...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    if os.path.exists(DB_PATH):
        db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        logger.info("✅ Векторная база FAISS загружена")
    else:
        st.error(f"⚠️ Критическая ошибка: База {DB_PATH} не найдена на сервере!")
        return None, site_index

    # Проверка базы шаблонов
    t_db_path = DB_PATH + "_templates"
    if os.path.exists(t_db_path):
        db_t = FAISS.load_local(t_db_path, embeddings, allow_dangerous_deserialization=True)
        db.merge_from(db_t)
        logger.info("✅ Шаблоны интегрированы в поиск")
        
    return db, site_index

db, site_index = load_resources()

# === 5. ФУНКЦИИ ГЛУБОКОГО ПОИСКА И ОЧИСТКИ (БЕЗ СОКРАЩЕНИЙ) ===

def clean_document_name(filename: str) -> str:
    """Превращает техническое имя файла в читаемое юридическое название"""
    name = urllib.parse.unquote(filename)
    name = re.sub(r'\.(docx?|pdf|txt)$', '', name, flags=re.IGNORECASE)
    # Разлепляем слова (ПоложениеОПорядке -> Положение О Порядке)
    name = re.sub(r'([а-яёa-z])([А-ЯЁA-Z])', r'\1 \2', name)
    # Убираем технические префиксы СДА
    name = re.sub(r'^[\d\s]*СДА\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'^[\d\.\-\s]+', '', name)
    # Форматируем даты
    name = re.sub(r'(?<!от\s)(\d{2})[-_.](\d{2})[-_.](\d{4})', r'от \1.\2.\3', name)
    return re.sub(r'\s+', ' ', name).strip()

def find_link_in_index(query: str) -> list:
    """Умный маршрутизатор: ищет глубокие ссылки на разделы сайта СДА"""
    q = query.lower()
    res = []
    
    # Глубокое сопоставление разделов (Sveden)
    mapping = {
        'document': (['документ', 'устав', 'лиценз', 'аккредитац', 'локальн', 'приказ', 'положен'], 'https://sdamp.ru/sveden/document/'),
        'paid': (['платн', 'оплат', 'договор', 'стоимост', 'квитанц'], 'https://sdamp.ru/sveden/paid_edu/'),
        'grants': (['стипенди', 'материальн', 'поддержк', 'пособи', 'выплат'], 'https://sdamp.ru/sveden/grants/'),
        'edu': (['расписани', 'календар', 'график', 'сесси', 'учебн'], 'https://sdamp.ru/sveden/education/'),
        'abitur': (['поступлен', 'прием', 'абитуриент', 'экзамен'], 'https://sdamp.ru/abitur/'),
        'struct': (['структур', 'руководств', 'ректорат', 'кафедр', 'деканат'], 'https://sdamp.ru/sveden/struct/'),
        'eios': (['эиос', 'личный кабинет', 'сдо', 'портал'], 'https://eios.sdamp.ru')
    }
    
    for key, (keywords, url) in mapping.items():
        if any(k in q for k in keywords):
            title = "Раздел: " + keywords[0].capitalize()
            res.append({'title': title, 'url': url})

    # Поиск по загруженному JSON-индексу
    for page in site_index.get('pages', []):
        if q in page.get('title', '').lower():
            res.append({'title': page.get('title'), 'url': page.get('url')})
            
    # Удаление дубликатов URL
    unique_res, seen = [], set()
    for item in res:
        if item['url'] not in seen:
            unique_res.append(item); seen.add(item['url'])
            
    return unique_res[:3]

def extract_keywords(query: str) -> list:
    stop_words = {'как', 'что', 'где', 'когда', 'почему', 'можно', 'нужно', 'могу', 'ли', 'или', 'подскаж'}
    words = re.findall(r'\b[а-яёА-ЯЁ]{4,}\b', query.lower())
    return [w for w in words if w not in stop_words]

def extract_document_references(docs: list) -> list:
    references = []
    patterns = [r'[Пп]оложение[а-яё\s]*["«]([^"]+)["»]', r'[Пп]риказ[а-яё\s]*№?\s*\d+.*["«]([^"]+)["»]']
    for doc in docs:
        for p in patterns: references.extend(re.findall(p, doc['content']))
    return list(set(references))[:10]

def iterative_search(query: str):
    """Твой оригинальный 3-х стадийный алгоритм поиска"""
    if not db: return [], set()
    found_docs, sources_set = [], set()
    
    # 1 стадия: Смысловой поиск (FAISS)
    docs_s1 = db.similarity_search(query, k=12)
    for d in docs_s1:
        s = clean_document_name(os.path.basename(d.metadata.get('source', 'Неизвестный')))
        sources_set.add(s)
        found_docs.append({'source': s, 'content': d.page_content, 'stage': 1})
    
    # 2 стадия: Поиск по выделенным ключевым словам
    for term in extract_keywords(query)[:3]:
        for d in db.similarity_search(term, k=5):
            s = clean_document_name(os.path.basename(d.metadata.get('source', 'Неизвестный')))
            if s not in [doc['source'] for doc in found_docs]:
                sources_set.add(s)
                found_docs.append({'source': s, 'content': d.page_content, 'stage': 2})
    
    # 3 стадия: Поиск по найденным ссылкам на другие акты
    for doc_ref in extract_document_references(found_docs)[:5]:
        for d in db.similarity_search(doc_ref, k=4):
            s = clean_document_name(os.path.basename(d.metadata.get('source', 'Неизвестный')))
            if s not in [doc['source'] for doc in found_docs]:
                sources_set.add(s)
                found_docs.append({'source': s, 'content': d.page_content, 'stage': 3})
                
    return found_docs, sources_set

def find_template(user_query: str) -> str | None:
    if not os.path.exists(TEMPLATES_PATH): return None
    templates = os.listdir(TEMPLATES_PATH)
    q = user_query.lower()
    # Расширенное сопоставление для шаблонов
    synonyms = {'академ':['академ', 'отпуск'], 'отчисл':['отчисл', 'забрат', 'выбыт'], 'стипенди':['стипенди'], 'справк':['справк']}
    for key, terms in synonyms.items():
        if any(t in q for t in terms):
            for f in templates:
                if any(t in f.lower() for t in terms): return os.path.join(TEMPLATES_PATH, f)
    return None

def parse_suggestions(answer: str) -> list:
    suggestions = []
    match = re.search(r'(🎯\s*УТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n?)(.+)', answer, re.IGNORECASE | re.DOTALL)
    if match:
        questions = re.findall(r'\[([^\]]+)\]|\b([А-Яа-яёЁ].*?\?)', match.group(2).strip())
        for q in questions:
            for part in q:
                if part.strip(): suggestions.append(part.strip())
    return suggestions[:3]

def clean_answer(answer: str) -> str:
    return re.sub(r'\n*🎯\s*УТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n?.*', '', answer, flags=re.DOTALL | re.IGNORECASE).strip()

# === 6. ГЛУБОКИЙ СИСТЕМНЫЙ ПРОМПТ И ВЫЗОВ ЯНДЕКСА (С ПАМЯТЬЮ) ===

DEEP_SYSTEM_PROMPT = """
Ты — Интеллектуальный юридический ассистент Сретенской духовной академии (СДА). 
Твоя роль — эксперт-методист, в совершенстве знающий локальную нормативную базу Академии и ФЗ-273 "Об образовании".

ТВОЯ ЗАДАЧА: Давать юридически безупречные ответы, опираясь ТОЛЬКО на предоставленные фрагменты документов.

ПРАВИЛА ОТВЕТА (КРИТИЧЕСКИ ВАЖНО):

1. СТРУКТУРА (СТРОГО):
📌 **ЗАКЛЮЧЕНИЕ** (1-2 предложения — суть ответа: разрешено, запрещено, возможно при условиях).
📖 **ПРАВОВОЕ ОБОСНОВАНИЕ** (детальный разбор ситуации со ссылками на конкретные ПУНКТЫ и СТАТЬИ документов). 
   *Пример: "На основании п. 5.1 Положения о текущем контроле..."*
📋 **ПОРЯДОК ДЕЙСТВИЙ** (пошаговый алгоритм для студента или сотрудника).
📎 **ДОКУМЕНТЫ** (полный перечень названий актов, которые были использованы).
🔗 **ССЫЛКИ НА САЙТ** (используй ТОЛЬКО те URL, которые переданы тебе в блоке "РЕЛЕВАНТНЫЕ ССЫЛКИ").

2. АНАЛИЗ: Если в документах есть противоречие или пробел — укажи на это и порекомендуй обратиться в Учебную часть.
3. ЗАПРЕТ ГАЛЛЮЦИНАЦИЙ: Не выдумывай названия приказов, даты и ссылки. Если данных нет — так и пиши.

4. УТОЧНЯЮЩИЕ ВОПРОСЫ:
   В самом конце ты ОБЯЗАН предложить 2-3 вопроса ОТ ЛИЦА СТУДЕНТА. Эти вопросы должны развивать тему (например, про сроки или необходимые документы).
   Формат: 🎯 УТОЧНЯЮЩИЕ ВОПРОСЫ: [Вопрос 1?] [Вопрос 2?]
"""

def call_yandex_gpt(history, current_question, context, site_links):
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {"Authorization": f"Api-Key {YANDEX_API_KEY}", "x-folder-id": FOLDER_ID}
    
    # Сборка сообщений с ПАМЯТЬЮ
    messages = [{"role": "system", "text": DEEP_SYSTEM_PROMPT}]
    for msg in history[-6:]: # Помним последние 6 реплик
        role = "assistant" if msg["role"] == "assistant" else "user"
        messages.append({"role": role, "text": msg["content"]})
    
    links_text = "\n".join([f"- {l['title']}: {l['url']}" for l in site_links])
    user_payload = f"КОНТЕКСТ ИЗ ДОКУМЕНТОВ:\n{context}\n\nРЕЛЕВАНТНЫЕ ССЫЛКИ НА САЙТЕ:\n{links_text}\n\nВОПРОС ПОЛЬЗОВАТЕЛЯ: {current_question}"
    messages.append({"role": "user", "text": user_payload})

    payload = {
        "modelUri": f"gpt://{FOLDER_ID}/yandexgpt/latest",
        "completionOptions": {"temperature": 0.2, "maxTokens": "2000"},
        "messages": messages
    }
    
    res = requests.post(url, headers=headers, json=payload)
    if res.status_code == 200: return res.json()['result']['alternatives'][0]['message']['text']
    else: raise Exception(f"Ошибка YandexGPT: {res.text}")

def get_rag_response(question: str, chat_history: list):
    docs, sources = iterative_search(question)
    if not docs: return "😔 В официальных документах Академии ответ на данный вопрос не найден. Рекомендую обратиться в Учебную часть СДА.", [], ""

    docs.sort(key=lambda x: x['stage'])
    context = "\n\n".join([f"--- ФРАГМЕНТ (Источник: {d['source']}) ---\n{d['content']}" for i, d in enumerate(docs[:15])])
    site_links = find_link_in_index(question)
    
    try:
        raw_answer = call_yandex_gpt(chat_history, question, context, site_links)
        sources_text = "\n".join([f"• {s}" for s in sources])
        suggestions = parse_suggestions(raw_answer)
        answer = clean_answer(raw_answer)
        return answer, suggestions, sources_text
    except Exception as e:
        logger.error(f"Ошибка ИИ: {e}")
        return f"⚠️ Произошла техническая ошибка при обращении к ИИ: {e}", [], ""

# === 7. ГЛАВНЫЙ ИНТЕРФЕЙС И ЛОГИКА ===

if "messages" not in st.session_state:
    welcome = (
        "👋 Здравствуйте! Я — Интеллектуальный помощник Сретенской духовной академии.\n"
        "Я помогу Вам найти информацию в Уставе, Положениях и Приказах Академии.\n\n"
        "📖 **Что я умею?**\n"
        "• Разъясняю правила обучения, перевода и отчисления\n"
        "• Помогаю найти глубокие ссылки на сайте sdamp.ru\n"
        "• Подбираю необходимые шаблоны заявлений"
    )
    st.session_state.messages = [{"role": "assistant", "content": welcome, "sources": None, "template": None, 
                                 "suggestions": ["Как оформить академический отпуск?", "Какие документы нужны для отчисления?", "Как получить справку об обучении?"]}]

# Рендеринг истории
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("template"):
            with open(msg["template"], "rb") as f:
                st.download_button("📥 Скачать документ", f.read(), file_name=os.path.basename(msg["template"]), key=f"dl_{i}_{uuid.uuid4()}")
        if msg.get("sources"):
            with st.expander("📚 Ссылки на первоисточники (база знаний)"): st.markdown(msg["sources"])
        
        # Кнопки подсказок (только для самого последнего сообщения ассистента)
        if msg.get("suggestions") and msg["role"] == "assistant" and i == len(st.session_state.messages)-1:
            st.markdown("💡 *Возможно, Вас также заинтересует:*")
            for sug in msg["suggestions"]:
                if st.button(sug, key=f"sug_{i}_{sug}"):
                    st.session_state.suggestion_clicked = sug; st.rerun()

# Ввод вопроса
prompt = st.chat_input("Задайте вопрос по нормативным актам...")

if "suggestion_clicked" in st.session_state:
    prompt = st.session_state.suggestion_clicked
    del st.session_state.suggestion_clicked

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Анализирую нормативную базу..."):
            # 1. Поиск шаблона
            t_path = find_template(prompt)
            if t_path:
                fname = os.path.basename(t_path)
                ans = f"📄 **Нашёл официальный шаблон:** {clean_document_name(fname)}\nПожалуйста, скачайте его, заполните и подайте в учебную часть."
                st.markdown(ans)
                with open(t_path, "rb") as f: 
                    st.download_button("📥 Скачать документ", f.read(), file_name=fname, key=f"dl_new_{uuid.uuid4()}")
                st.session_state.messages.append({"role": "assistant", "content": ans, "template": t_path})
                save_message(st.session_state.user_id, prompt, ans); st.rerun()
            else:
                # 2. RAG Поиск
                answer, suggestions, sources_text = get_rag_response(prompt, st.session_state.messages[:-1])
                st.markdown(answer)
                if sources_text:
                    with st.expander("📚 Ссылки на первоисточники (база знаний)"): st.markdown(sources_text)
                
                # Кнопки полезности (фидбек)
                c1, c2, c3 = st.columns([1,1,4])
                with c1: 
                    if st.button("👍 Полезно", key=f"ok_{uuid.uuid4()}"): save_feedback(st.session_state.user_id, prompt, True)
                with c2: 
                    if st.button("👎 Нет", key=f"no_{uuid.uuid4()}"): save_feedback(st.session_state.user_id, prompt, False)
                
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources_text, "suggestions": suggestions})
                save_message(st.session_state.user_id, prompt, answer); st.rerun()
