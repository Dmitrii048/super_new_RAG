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
from bs4 import BeautifulSoup

# Импорты RAG
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# === 1. НАСТРОЙКИ И СЕКРЕТЫ ===
try:
    YANDEX_API_KEY = st.secrets["YANDEX_API_KEY"]
except:
    YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")

FOLDER_ID = os.getenv("FOLDER_ID", "b1g6jhk9eapudn6lom6c")

DB_PATH = os.getenv("DB_PATH", "sretensk_db")
TEMPLATES_PATH = os.getenv("TEMPLATES_PATH", "docs/templates")
SITE_INDEX_FILE = "docs/site_index.json"
DB_FILE = "chat_history.db"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === 2. НАСТРОЙКА ИНТЕРФЕЙСА (АКАДЕМИЧЕСКИЙ ДИЗАЙН) ===
st.set_page_config(page_title="Юридический ассистент СДА", page_icon="🎓", layout="centered")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 900px;
    }
    
    .stApp { background-color: #FCFCFA; }
    
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
    
    .stChatMessage { border-radius: 12px; padding: 15px; border: 1px solid #eaeaea; margin-bottom: 10px; }
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
    except Exception as e: logger.error(f"Ошибка БД (сообщения): {e}")

def save_feedback(user_id, question, is_positive):
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute('INSERT INTO feedback (user_id, question, is_positive) VALUES (?, ?, ?)', (user_id, question, 1 if is_positive else 0))
        conn.commit()
        conn.close()
    except Exception as e: logger.error(f"Ошибка БД (оценки): {e}")

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
    else:
        st.error(f"⚠️ Критическая ошибка: База {DB_PATH} не найдена на сервере!")
        return None, site_index

    t_db_path = DB_PATH + "_templates"
    if os.path.exists(t_db_path):
        db_t = FAISS.load_local(t_db_path, embeddings, allow_dangerous_deserialization=True)
        db.merge_from(db_t)
        
    return db, site_index

db, site_index = load_resources()

# === 5. ПАРСИНГ САЙТА В РЕАЛЬНОМ ВРЕМЕНИ ===
def scrape_website_content(url: str) -> str:
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=5)
        response.encoding = 'utf-8'
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style", "header", "footer", "nav"]):
                script.extract()
            text = soup.get_text(separator=' ', strip=True)
            return text[:3500] 
    except Exception as e:
        logger.error(f"Ошибка парсинга {url}: {e}")
    return ""

# === 6. ЛОГИКА ПОИСКА И ОЧИСТКИ ТЕКСТОВ ===
def clean_document_name(filename: str) -> str:
    name = urllib.parse.unquote(filename)
    name = re.sub(r'\.(docx?|pdf|txt)$', '', name, flags=re.IGNORECASE)
    name = name.replace('_', ' ').replace('-', ' ')
    name = re.sub(r'([а-яёa-z])([А-ЯЁA-Z])', r'\1 \2', name)
    name = re.sub(r'^[\d\s]*СДА\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'^[\d\.\s]+', '', name)
    name = re.sub(r'ДОПОЛНЕНО.*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'Журнал.*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'(?<!от\s)(\d{2})\s(\d{2})\s(\d{4})', r'от \1.\2.\3', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name.capitalize() if name else "Нормативный документ СДА"

def find_link_in_index(query: str) -> list:
    q = query.lower()
    res =[]
    
    mapping = {
        'document': (['документ', 'устав', 'лиценз', 'аккредитац', 'локальн', 'приказ', 'положен'], 'https://sdamp.ru/sveden/document/'),
        'paid': (['платн', 'оплат', 'договор', 'стоимост', 'квитанц'], 'https://sdamp.ru/sveden/paid_edu/'),
        'grants': (['стипенди', 'материальн', 'поддержк', 'пособи', 'выплат'], 'https://sdamp.ru/sveden/grants/'),
        'edu': (['расписани', 'календар', 'график', 'сесси', 'учебн'], 'https://sdamp.ru/sveden/education/'),
        'abitur': (['поступлен', 'прием', 'абитуриент', 'экзамен', 'возраст', 'поступит'], 'https://sdamp.ru/abitur/'),
        'struct': (['структур', 'руководств', 'ректорат', 'кафедр', 'деканат'], 'https://sdamp.ru/sveden/struct/'),
        'eios': (['эиос', 'личный кабинет', 'сдо', 'портал', 'пароль'], 'https://eios.sdamp.ru')
    }
    
    for key, (keywords, url) in mapping.items():
        if any(k in q for k in keywords):
            res.append({'title': 'Перейти в раздел: ' + keywords[0].capitalize(), 'url': url})

    for page in site_index.get('pages',[]):
        if q in page.get('title', '').lower():
            res.append({'title': page.get('title'), 'url': page.get('url')})
            
    unique_res, seen =[], set()
    for item in res:
        if item['url'] not in seen:
            unique_res.append(item); seen.add(item['url'])
            
    return unique_res[:3]

def extract_keywords(query: str) -> list:
    stop_words = {'как', 'что', 'где', 'когда', 'почему', 'можно', 'нужно', 'могу', 'ли', 'или', 'подскаж'}
    words = re.findall(r'\b[а-яёА-ЯЁ]{4,}\b', query.lower())
    return[w for w in words if w not in stop_words]

def extract_document_references(docs: list) -> list:
    references =[]
    patterns = [r'[Пп]оложение[а-яё\s]*["«]([^"]+)["»]', r'[Пп]риказ[а-яё\s]*№?\s*\d+.*["«]([^"]+)["»]']
    for doc in docs:
        for p in patterns: references.extend(re.findall(p, doc['content']))
    return list(set(references))[:10]

def iterative_search(query: str):
    if not db: return[], set()
    found_docs, sources_set =[], set()
    
    docs_s1 = db.similarity_search(query, k=12)
    for d in docs_s1:
        s = clean_document_name(os.path.basename(d.metadata.get('source', 'Неизвестный')))
        sources_set.add(s)
        found_docs.append({'source': s, 'content': d.page_content, 'stage': 1})
    
    for term in extract_keywords(query)[:3]:
        for d in db.similarity_search(term, k=5):
            s = clean_document_name(os.path.basename(d.metadata.get('source', 'Неизвестный')))
            if s not in [doc['source'] for doc in found_docs]:
                sources_set.add(s)
                found_docs.append({'source': s, 'content': d.page_content, 'stage': 2})
    
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
    synonyms = {
        'академ':['академ', 'отпуск'], 
        'отчисл':['отчисл', 'забрат', 'выбыт'], 
        'стипенди':['стипенди'], 
        'справк':['справк', 'архив']
    }
    for key, terms in synonyms.items():
        if any(t in q for t in terms):
            for f in templates:
                if any(t in f.lower() for t in terms): return os.path.join(TEMPLATES_PATH, f)
    return None

def parse_suggestions(answer: str) -> list:
    """Бронебойный парсер: вытаскивает вопросы в формате кнопок, как бы ИИ их ни написал"""
    suggestions =[]
    
    # Пытаемся найти блок "УТОЧНЯЮЩИЕ ВОПРОСЫ"
    match = re.search(r'УТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n*(.+)', answer, re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip()
        lines = text.split('\n')
        for line in lines:
            # Вычищаем цифры (1. 2.), маркеры и скобки
            clean_line = re.sub(r'^[\d\.\)\-\*\[\]\s]+', '', line).strip()
            clean_line = re.sub(r'\]$', '', clean_line).strip()
            if clean_line and clean_line.endswith('?'):
                suggestions.append(clean_line)
                
    # Фолбэк: если блок не найден, просто ищем вопросительные предложения в конце ответа
    if not suggestions:
        lines = answer.split('\n')
        for line in reversed(lines[-7:]): # Смотрим последние 7 строк
            if '?' in line:
                clean_line = re.sub(r'^[\d\.\)\-\*\[\]\s]+', '', line).strip()
                clean_line = re.sub(r'\]$', '', clean_line).strip()
                if clean_line and clean_line.endswith('?'):
                    if clean_line not in suggestions:
                        suggestions.insert(0, clean_line)
                        
    return suggestions[:3]

def clean_answer(answer: str) -> str:
    """Вырезает из текста ответы блок вопросов, так как они становятся интерактивными кнопками"""
    ans = re.sub(r'\n*(🎯|💡)?\s*УТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*.*', '', answer, flags=re.DOTALL | re.IGNORECASE)
    return ans.strip()


# === 7. ГЛУБОКИЙ СИСТЕМНЫЙ ПРОМПТ И YANDEX GPT ===
DEEP_SYSTEM_PROMPT = """
Ты — Интеллектуальный юридический ассистент Сретенской духовной академии (СДА). 
Твоя роль — вежливый, эрудированный и внимательный эксперт-методист. Ты в совершенстве знаешь локальную нормативную базу Академии и ФЗ-273 "Об образовании".

ТВОЯ ЗАДАЧА: Давать исчерпывающие, юридически грамотные ответы, опираясь ТОЛЬКО на предоставленные фрагменты документов и актуальные данные с сайта.

ПРАВИЛА ФОРМИРОВАНИЯ ОТВЕТА (КРИТИЧЕСКИ ВАЖНО):

1. СТРУКТУРА ОТВЕТА (Соблюдать жестко):
📌 **ЗАКЛЮЧЕНИЕ** (1-2 предложения — суть ответа: разрешено, запрещено, возможно при условиях).
📖 **ПРАВОВОЕ ОБОСНОВАНИЕ** (Детальный разбор ситуации. ОБЯЗАТЕЛЬНО ссылайся на конкретные пункты и статьи предоставленных документов). 
📋 **ПОРЯДОК ДЕЙСТВИЙ** (Если применимо — напиши пошаговый алгоритм действий для студента).
📎 **ДОКУМЕНТЫ** (Краткий перечень актов, которые были использованы).

2. АНАЛИЗ И КРИТИЧЕСКОЕ МЫШЛЕНИЕ:
- Внимательно читай КОНТЕКСТ. Если в локальных актах нет прямого ответа, но есть информация из блока "АКТУАЛЬНАЯ ИНФОРМАЦИЯ С САЙТА", используй её.
- Если ответа нет нигде — НЕ ВЫДУМЫВАЙ. Честно напиши: "К сожалению, в доступной мне нормативной базе СДА нет точного ответа на этот вопрос. Рекомендую обратиться лично в Учебную часть".

3. УТОЧНЯЮЩИЕ ВОПРОСЫ (ЭТО ОЧЕНЬ ВАЖНО):
В самом конце своего ответа ты ОБЯЗАН предложить 2-3 вопроса. 
ОНИ ДОЛЖНЫ БЫТЬ СФОРМУЛИРОВАНЫ ОТ ЛИЦА СТУДЕНТА (пользователя), как будто он хочет задать тебе следующий вопрос, чтобы углубиться в тему.
НЕ ЗАДАВАЙ вопросы пользователю (не пиши "Вам выслать бланк?"). ПРЕДЛАГАЙ ему готовые варианты!
Формат вывода строго такой:
УТОЧНЯЮЩИЕ ВОПРОСЫ:
1. [Вопрос 1?]
2.[Вопрос 2?]
"""

def call_yandex_gpt(history, current_question, context, site_links):
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {"Authorization": f"Api-Key {YANDEX_API_KEY}", "x-folder-id": FOLDER_ID}
    
    # Сборка сообщений с ПАМЯТЬЮ
    messages =[{"role": "system", "text": DEEP_SYSTEM_PROMPT}]
    for msg in history[-6:]: 
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
    site_links = find_link_in_index(question)
    
    # 1. LIVE WEB SCRAPING
    live_web_context = ""
    q_low = question.lower()
    if any(word in q_low for word in['поступ', 'абитуриент', 'возраст', 'экзамен']):
        web_text = scrape_website_content('https://sdamp.ru/abitur/')
        if web_text: live_web_context = f"\n--- АКТУАЛЬНАЯ ИНФОРМАЦИЯ С САЙТА (Абитуриенту) ---\n{web_text}"
    elif any(word in q_low for word in ['стипенд', 'выплат', 'пособи']):
        web_text = scrape_website_content('https://sdamp.ru/sveden/grants/')
        if web_text: live_web_context = f"\n--- АКТУАЛЬНАЯ ИНФОРМАЦИЯ С САЙТА (Стипендии) ---\n{web_text}"

    if not docs and not live_web_context: 
        return "😔 В официальных документах и на сайте Академии ответ на данный вопрос не найден. Рекомендую обратиться в Учебную часть СДА.",[], ""

    docs.sort(key=lambda x: x['stage'])
    context = "\n\n".join([f"--- ФРАГМЕНТ (Источник: {d['source']}) ---\n{d['content']}" for i, d in enumerate(docs[:15])])
    
    # Объединяем локальную базу и сайт
    full_context = context + live_web_context
    
    try:
        # Вызов ИИ
        raw_answer = call_yandex_gpt(chat_history, question, full_context, site_links)
        
        # ФОРМИРОВАНИЕ ПРЯМЫХ ГИПЕРССЫЛОК НА ДОКУМЕНТЫ
        clean_sources =[]
        for s in sources:
            if not s.strip(): continue
            doc_url = None
            # Пытаемся найти точную ссылку на PDF файл в индексе сайта
            for doc in site_index.get('documents',[]):
                if s.lower() in doc.get('name', '').lower() or doc.get('name', '').lower() in s.lower():
                    doc_url = doc.get('url')
                    break
            
            if doc_url:
                clean_sources.append(f"📄 [{s}]({doc_url})") # Прямая ссылка на PDF
            else:
                clean_sources.append(f"📄 {s}") # Просто название, если ссылки нет
        
        sources_text = "\n\n".join(clean_sources)
        
        if site_links:
            links_text = "\n".join([f"🔗 [{link['title']}]({link['url']})" for link in site_links])
            sources_text += f"\n\n🌐 **Полезные разделы сайта:**\n{links_text}"
            
        suggestions = parse_suggestions(raw_answer)
        answer = clean_answer(raw_answer)
        
        # Защита от галлюцинаций про возраст
        if "возраст" in question.lower() or "лет" in question.lower():
            if "60" in answer or "ограничений нет" in answer.lower():
                answer += "\n\n⚠️ *Примечание методиста: Обратите внимание, что по актуальным правилам приема возраст абитуриентов, поступающих на бакалавриат, ограничен 35 годами (для очного) и 50 годами (для заочного).* (Уточните на сайте)."
        
        return answer, suggestions, sources_text
    except Exception as e:
        logger.error(f"Ошибка ИИ: {e}")
        return f"⚠️ Произошла техническая ошибка при обращении к ИИ: {e}",[], ""


# === 8. ГЛАВНЫЙ ИНТЕРФЕЙС ===

if "messages" not in st.session_state:
    welcome = (
        "Здравствуйте! Я с радостью отвечу на ваши вопросы, касающиеся учебного процесса, правил поступления, перевода или отчисления.\n\n"
        "Если вам потребуется найти конкретный нормативный документ, бланк заявления или нужный раздел на нашем официальном сайте — просто спросите меня об этом!"
    )
    st.session_state.messages =[{"role": "assistant", "content": welcome, "sources": None, "template": None, 
                                 "suggestions":["Какие документы нужны для перевода на бюджет?", "Как мне оформить академический отпуск?", "До какого возраста принимают в Академию?"]}]

# Рендеринг истории сообщений
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Отрисовка кнопки скачивания шаблона
        if msg.get("template") and os.path.exists(msg["template"]):
            with open(msg["template"], "rb") as f:
                st.download_button("📥 Скачать шаблон документа", f.read(), file_name=os.path.basename(msg["template"]), key=f"dl_{i}_{uuid.uuid4()}")
                
        # Отрисовка источников
        if msg.get("sources"):
            with st.expander("📚 Ссылки на официальные документы"): 
                st.markdown(msg["sources"])
        
        # Интерактивные вопросы (только для последнего ответа)
        if msg.get("suggestions") and msg["role"] == "assistant" and i == len(st.session_state.messages)-1:
            st.markdown("💡 *Возможно, вас заинтересует:*")
            for sug in msg["suggestions"]:
                if st.button(sug, key=f"sug_{i}_{sug}"):
                    st.session_state.suggestion_clicked = sug; st.rerun()

# Ввод нового вопроса
prompt = st.chat_input("Спросите о документах или правилах Академии...")

if "suggestion_clicked" in st.session_state:
    prompt = st.session_state.suggestion_clicked
    del st.session_state.suggestion_clicked

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): 
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Анализирую базу знаний СДА..."):
            
            # Проверка: вдруг пользователь просит шаблон?
            t_path = find_template(prompt)
            if t_path:
                fname = os.path.basename(t_path)
                ans = f"📄 **Подготовил для вас официальный бланк:** {clean_document_name(fname)}\nПожалуйста, скачайте его, заполните и направьте в Учебную часть."
                st.markdown(ans)
                with open(t_path, "rb") as f: 
                    st.download_button("📥 Скачать шаблон документа", f.read(), file_name=fname, key=f"dl_new_{uuid.uuid4()}")
                
                st.session_state.messages.append({"role": "assistant", "content": ans, "template": t_path})
                save_message(st.session_state.user_id, prompt, ans); st.rerun()
                
            else:
                # Запускаем мощный RAG-конвейер
                answer, suggestions, sources_text = get_rag_response(prompt, st.session_state.messages[:-1])
                
                st.markdown(answer)
                if sources_text:
                    with st.expander("📚 Ссылки на официальные документы"): 
                        st.markdown(sources_text)
                
                # Кнопки оценки работы
                c1, c2, c3 = st.columns([1,1,4])
                with c1: 
                    if st.button("👍 Полезно", key=f"ok_{uuid.uuid4()}"): save_feedback(st.session_state.user_id, prompt, True)
                with c2: 
                    if st.button("👎 Нет", key=f"no_{uuid.uuid4()}"): save_feedback(st.session_state.user_id, prompt, False)
                
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources_text, "suggestions": suggestions})
                save_message(st.session_state.user_id, prompt, answer); st.rerun()
