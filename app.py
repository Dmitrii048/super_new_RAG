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

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# === 1. НАСТРОЙКИ И СЕКРЕТЫ ===
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

# === 2. НАСТРОЙКА ИНТЕРФЕЙСА СТРИМЛИТ (ПЛОТНЫЙ ДИЗАЙН СДА) ===
st.set_page_config(page_title="Юридический ассистент СДА", page_icon="🎓", layout="centered")

st.markdown("""
    <style>
    /* Убираем стандартные элементы Streamlit */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* УБИРАЕМ ПУСТЫЕ МЕСТА СВЕРХУ И СНИЗУ */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 900px;
    }
    
    .stApp { background-color: #FCFCFA; }
    
    /* Дизайн баннера СДА */
    .sda-banner {
        background: linear-gradient(90deg, rgba(255,255,255,1) 0%, rgba(255,255,255,0.95) 50%, rgba(255,255,255,0) 100%), 
                    url('https://sdamp.ru/bitrix/templates/main/img/header/day_spring.png');
        background-size: cover;
        background-position: right center;
        border-left: 8px solid #942927; 
        border-radius: 12px;
        padding: 25px 30px;
        display: flex;
        align-items: center;
        gap: 20px;
        margin-top: -20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    
    .sda-logo { width: 90px; height: auto; flex-shrink: 0; }
    .sda-text-block { display: flex; flex-direction: column; }
    
    .sda-title {
        font-family: 'Times New Roman', Times, serif;
        color: #1a2a44; 
        font-size: 24px;
        font-weight: 900;
        margin: 0;
        line-height: 1.1;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .sda-subtitle {
        font-family: 'Arial', sans-serif;
        color: #942927; 
        font-size: 14px;
        margin-top: 8px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Стилизация сообщений */
    .stChatMessage { border-radius: 12px; padding: 15px; border: 1px solid #eaeaea; }
    .stButton>button { background-color: #f0f2f6; border-radius: 20px; border: 1px solid #ddd; color: #1a2a44; }
    .stButton>button:hover { background-color: #e2e6ea; border-color: #c5a059; color: #942927; }
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

# === 3. БАЗА ДАННЫХ SQLITE ===
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
    except Exception as e: logger.error(f"Ошибка БД: {e}")

def save_feedback(user_id, question, is_positive):
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute('INSERT INTO feedback (user_id, question, is_positive) VALUES (?, ?, ?)', (user_id, question, 1 if is_positive else 0))
        conn.commit()
        conn.close()
    except Exception as e: logger.error(f"Ошибка БД: {e}")

# === 4. ЗАГРУЗКА РЕСУРСОВ ===
@st.cache_resource
def load_resources():
    site_index = {'pages':[], 'documents':[]}
    if os.path.exists(SITE_INDEX_FILE):
        try:
            with open(SITE_INDEX_FILE, 'r', encoding='utf-8') as f: site_index = json.load(f)
        except: pass

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    if os.path.exists(DB_PATH):
        db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        st.error("⚠️ База данных sretensk_db не найдена в репозитории!")
        return None, site_index

    templates_db_path = DB_PATH + "_templates"
    if os.path.exists(templates_db_path):
        db_templates = FAISS.load_local(templates_db_path, embeddings, allow_dangerous_deserialization=True)
        db.merge_from(db_templates)
        
    return db, site_index

db, site_index = load_resources()

# === 5. ЛОГИКА ПОИСКА (3 СТАДИИ) ===
def clean_document_name(filename: str) -> str:
    """Улучшение: очистка технических имен файлов для красивого вывода"""
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
    results =[]
    for page in site_index.get('pages',[]):
        if query_lower in page.get('title', '').lower() or query_lower in page.get('url', '').lower():
            results.append({'title': page.get('title', 'Страница'), 'url': page.get('url', '')})
    for doc in site_index.get('documents',[]):
        if query_lower in doc.get('name', '').lower():
            results.append({'title': doc.get('name', 'Документ'), 'url': doc.get('url', '')})
    return results[:5]

def extract_keywords(query: str) -> list:
    stop_words = {'как', 'что', 'где', 'когда', 'почему', 'можно', 'нужно', 'могу', 'ли', 'или', 'и', 'в', 'на', 'по', 'для', 'при', 'о', 'об'}
    words = re.findall(r'\b[а-яёА-ЯЁ]{4,}\b', query.lower())
    return[w for w in words if w not in stop_words]

def extract_document_references(docs: list) -> list:
    references = []
    patterns = [r'[Пп]оложение[а-яё\s]*["«]([^"]+)["»]', r'[Пп]риказ[а-яё\s]*№?\s*\d+.*["«]([^"]+)["»]']
    for doc in docs:
        for pattern in patterns: references.extend(re.findall(pattern, doc['content']))
    return list(set(references))[:10]

def iterative_search(query: str):
    if not db: return[], set()
    found_docs =[]
    sources_set = set()
    
    # 1 стадия
    docs_stage1 = db.similarity_search(query, k=12)
    for d in docs_stage1:
        source = clean_document_name(os.path.basename(d.metadata.get('source', 'Неизвестный')))
        sources_set.add(source)
        found_docs.append({'source': source, 'content': d.page_content, 'stage': 1})
    
    # 2 стадия
    for term in extract_keywords(query)[:3]:
        docs_stage2 = db.similarity_search(term, k=6)
        for d in docs_stage2:
            source = clean_document_name(os.path.basename(d.metadata.get('source', 'Неизвестный')))
            if source not in [doc['source'] for doc in found_docs]:
                sources_set.add(source)
                found_docs.append({'source': source, 'content': d.page_content, 'stage': 2})
                
    # 3 стадия
    for doc_ref in extract_document_references(found_docs)[:5]:
        docs_stage3 = db.similarity_search(doc_ref, k=4)
        for d in docs_stage3:
            source = clean_document_name(os.path.basename(d.metadata.get('source', 'Неизвестный')))
            if source not in [doc['source'] for doc in found_docs]:
                sources_set.add(source)
                found_docs.append({'source': source, 'content': d.page_content, 'stage': 3})
                
    return found_docs, sources_set

def find_template(user_query: str) -> str | None:
    if not os.path.exists(TEMPLATES_PATH): return None
    templates = os.listdir(TEMPLATES_PATH)
    query_lower = user_query.lower()
    keywords_map = {
        'академ':['академ', 'отпуск', 'приостановл'],
        'отчисл':['отчисл', 'выбыт', 'забрат', 'исключ'],
        'пересдач':['пересдач', 'оценк', 'комисси'],
        'дистан':['дистан', 'онлайн', 'удален'],
        'справк':['справк', 'обучени', 'архив'],
        'общежити':['общежити', 'жиль', 'проживани'],
        'стипенди':['стипенди', 'выплат', 'матпомощ']
    }
    for _, search_terms in keywords_map.items():
        if any(term in query_lower for term in search_terms):
            for t in templates:
                if any(term in t.lower() for term in search_terms):
                    return os.path.join(TEMPLATES_PATH, t)
    return None

def parse_suggestions(answer: str) -> list:
    suggestions = []
    patterns =[r'🎯\s*УТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n?(.+)', r'УТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n?(.+)', r'💡\s*УТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n?(.+)']
    for pattern in patterns:
        match = re.search(pattern, answer, re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1).strip()
            questions = re.findall(r'\[([^\]]+)\]|\b([А-Яа-яёЁ].*?\?)', text)
            for q in questions:
                if isinstance(q, tuple):
                    for part in q:
                        if part.strip(): suggestions.append(part.strip())
                elif q.strip(): suggestions.append(q.strip())
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
    patterns =[r'\n*🎯\s*УТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n?.*', r'\n*УТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n?.*', r'\n*💡\s*УТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n?.*']
    for pattern in patterns:
        answer = re.sub(pattern, '', answer, flags=re.DOTALL | re.IGNORECASE)
    return answer.strip()

# === 6. СИСТЕМНЫЙ ПРОМПТ И ВЫЗОВ YANDEX GPT (С ПАМЯТЬЮ!) ===

# ТВОЙ ОРИГИНАЛЬНЫЙ ПРОМПТ + УТОЧНЕНИЕ ДЛЯ ВОПРОСОВ
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
📖 **ПРАВОВОЕ ОБОСНОВАНИЕ** [Развернутый анализ со ссылками на конкретные пункты]
📋 **ПОРЯДОК ДЕЙСТВИЙ** (если применимо)
📎 **ДОКУМЕНТЫ** •[Название документа, номер, пункт]
🔗 **ССЫЛКИ НА САЙТ** •[Проверенная ссылка на страницу сайта sdamp.ru, если релевантно]

2. ГЛУБОКИЙ ПОИСК: Анализируй ВСЕ найденные фрагменты
3. ПРОВЕРКА ССЫЛОК: НЕ выдумывай ссылки — используй ТОЛЬКО те, что найдены в контексте или индексе сайта.
4. ВАЖНО: Используй ТОЛЬКО информацию из документов. Не давай общих советов. НЕ галлюцинируй.

5. УТОЧНЯЮЩИЕ ВОПРОСЫ (КРИТИЧЕСКИ ВАЖНО!):
   В КОНЦЕ каждого ответа добавь 2-3 вопроса. 
   Эти вопросы должны быть сформулированы ОТ ЛИЦА СТУДЕНТА (пользователя), как будто он хочет задать тебе следующий вопрос. НЕ задавай вопросы пользователю (например, "Вам нужен файл?"), а ПРЕДЛАГАЙ ему готовые варианты вопросов, которые он может нажать!
   
   Формат вывода строго:
   🎯 УТОЧНЯЮЩИЕ ВОПРОСЫ:
   [Вопрос 1?][Вопрос 2?] [Вопрос 3?]
   
   Правильные примеры:
   [Как мне забрать документы?] [В какие сроки подается прошение?] [Какие виды стипендий бывают?]

Стиль: Профессиональный юридический, но доброжелательный.
"""

def call_yandex_gpt(history, current_question, context):
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {"Authorization": f"Api-Key {YANDEX_API_KEY}", "x-folder-id": FOLDER_ID}
    
    # 1. Сначала системный промпт
    yandex_messages =[{"role": "system", "text": SYSTEM_PROMPT}]
    
    # 2. ПАМЯТЬ: Добавляем последние 4 сообщения из истории диалога
    for msg in history[-4:]:
        if msg["role"] in ["user", "assistant"]:
            yandex_messages.append({"role": msg["role"], "text": msg["content"]})
            
    # 3. Добавляем текущий вопрос пользователя ВМЕСТЕ с документами (контекстом RAG)
    final_user_text = f"КОНТЕКСТ ДОКУМЕНТОВ:\n{context}\n\nМОЙ ТЕКУЩИЙ ВОПРОС: {current_question}\n\nДай структурированный ответ с уточняющими вопросами в конце. Проверь, что все ссылки реальны!"
    yandex_messages.append({"role": "user", "text": final_user_text})

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

def get_rag_response(question: str, chat_history: list):
    docs, sources = iterative_search(question)
    site_links = find_link_in_index(question)
    
    site_context = ""
    if site_links:
        site_context = "\n📎 РЕЛЕВАНТНЫЕ ССЫЛКИ НА САЙТЕ:\n"
        for link in site_links: site_context += f"- {link['title']}: {link['url']}\n"
            
    if not docs:
        if site_links:
            links_text = "\n".join([f"🔗 [{link['title']}]({link['url']})" for link in site_links])
            return f"В базе документов не найдено, но есть информация на сайте:\n{links_text}",[], ""
        return "😔 В базе знаний не найдено релевантных документов.", [], ""

    docs.sort(key=lambda x: x['stage'])
    context = "\n\n".join([f"--- ФРАГМЕНТ {i+1} ({d['source']}) ---\n{d['content']}" for i, d in enumerate(docs[:15])])
    if site_context: context += site_context
    
    try:
        # ПЕРЕДАЕМ ИСТОРИЮ ЧАТА В ИИ!
        raw_answer = call_yandex_gpt(chat_history, question, context)
        
        sources_text = "\n".join([f"• {s}" for s in sources])
        if site_links:
            links_text = "\n".join([f"🔗 [{link['title']}]({link['url']})" for link in site_links])
            sources_text += f"\n\n🌐 *Проверенные ссылки на сайте:*\n{links_text}"
            
        suggestions = parse_suggestions(raw_answer)
        answer = clean_answer(raw_answer)
        return answer, suggestions, sources_text
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        return f"Произошла ошибка при обращении к ИИ: {e}",[], ""


# === 7. ГЛАВНЫЙ ИНТЕРФЕЙС И ЛОГИКА ЧАТА ===
if "messages" not in st.session_state:
    welcome_text = (
        "👋 Здравствуйте! Я — Интеллектуальный помощник Сретенской духовной академии.\n"
        "Я знаю всё о Положениях, Приказах и Уставе.\n\n"
        "📖 **Чем я могу помочь?**\n"
        "• Отвечу на вопросы о правилах\n"
        "• Помогу найти нужный документ\n"
        "• Подберу шаблон заявления"
    )
    st.session_state.messages =[
        {"role": "assistant", "content": welcome_text, "sources": None, "template": None, 
         "suggestions":["Как оформить академический отпуск?", "Какие документы нужны для отчисления?", "Как получить справку об обучении?"]}
    ]

# Отрисовка истории сообщений
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # КНОПКА СКАЧИВАНИЯ ФАЙЛА DOCX (ИСПРАВЛЕНО)
        if msg.get("template") and os.path.exists(msg["template"]):
            with open(msg["template"], "rb") as f:
                file_bytes = f.read()
            st.download_button(
                label="📥 Скачать документ", 
                data=file_bytes, 
                file_name=os.path.basename(msg["template"]), 
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key=f"dl_{i}_{uuid.uuid4()}"
            )
                
        # Источники
        if msg.get("sources"):
            with st.expander("📚 Использованные документы"):
                st.markdown(msg["sources"])
        
        # Кнопки-подсказки (только у последнего ответа)
        if msg.get("suggestions") and msg["role"] == "assistant" and i == len(st.session_state.messages) - 1:
            st.markdown("💡 *Возможно, вас также заинтересует:*")
            for sug in msg["suggestions"]:
                if st.button(sug, key=f"sug_{i}_{sug}"):
                    st.session_state.suggestion_clicked = sug
                    st.rerun()

prompt = st.chat_input("Напишите ваш вопрос (например, 'Как оформить академ?')...")

if "suggestion_clicked" in st.session_state:
    prompt = st.session_state.suggestion_clicked
    del st.session_state.suggestion_clicked

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Изучаю нормативные акты СДА..."):
            
            # Проверяем, не нужен ли шаблон
            template_path = find_template(prompt)
            
            if template_path:
                filename = os.path.basename(template_path)
                ans_text = f"📄 **Нашёл подходящий шаблон:** {clean_document_name(filename)}\nПожалуйста, скачайте его, заполните и передайте в учебную часть."
                st.markdown(ans_text)
                
                # Кнопка скачивания
                try:
                    with open(template_path, "rb") as f:
                        file_bytes = f.read()
                    st.download_button(
                        label="📥 Скачать документ", 
                        data=file_bytes, 
                        file_name=filename, 
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key=f"dl_new_{uuid.uuid4()}"
                    )
                except Exception as e:
                    st.error(f"Не удалось загрузить файл: {e}")
                
                st.session_state.messages.append({"role": "assistant", "content": ans_text, "template": template_path})
                save_message(st.session_state.user_id, prompt, ans_text)
                st.rerun()
                
            else:
                # ОБРАЩЕНИЕ К ИИ (с передачей истории)
                answer, suggestions, sources_text = get_rag_response(prompt, st.session_state.messages[:-1])
                
                st.markdown(answer)
                if sources_text:
                    with st.expander("📚 Использованные документы"):
                        st.markdown(sources_text)
                
                col1, col2, col3 = st.columns([1, 1, 4])
                with col1:
                    if st.button("👍 Полезно", key=f"up_{uuid.uuid4()}"):
                        save_feedback(st.session_state.user_id, prompt, True)
                with col2:
                    if st.button("👎 Нет", key=f"down_{uuid.uuid4()}"):
                        save_feedback(st.session_state.user_id, prompt, False)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer, 
                    "sources": sources_text,
                    "suggestions": suggestions
                })
                
                full_log = f"{answer}\n\nИсточники:\n{sources_text}"
                save_message(st.session_state.user_id, prompt, full_log)
                st.rerun()
