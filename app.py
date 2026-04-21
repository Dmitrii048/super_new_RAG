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

# ИСПРАВЛЕННЫЙ FOLDER ID (Рабочий)
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
    /* Скрываем технические элементы Streamlit */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Жестко убираем пустые места сверху и снизу страницы */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
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
    
    /* Стилизация сообщений чата */
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

# Идентификатор пользователя для логов
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

# === 3. БАЗА ДАННЫХ SQLITE ДЛЯ ИСТОРИИ ===
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

# === 4. ЗАГРУЗКА БАЗЫ ЗНАНИЙ ===
@st.cache_resource
def load_resources():
    site_index = {'pages':[], 'documents':[]}
    if os.path.exists(SITE_INDEX_FILE):
        try:
            with open(SITE_INDEX_FILE, 'r', encoding='utf-8') as f:
                site_index = json.load(f)
        except: pass

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    if os.path.exists(DB_PATH):
        db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        st.error(f"⚠️ Критическая ошибка: База знаний '{DB_PATH}' не найдена!")
        return None, site_index

    t_db_path = DB_PATH + "_templates"
    if os.path.exists(t_db_path):
        db_t = FAISS.load_local(t_db_path, embeddings, allow_dangerous_deserialization=True)
        db.merge_from(db_t)
        
    return db, site_index

db, site_index = load_resources()

# === 5. ПАРСИНГ САЙТА В РЕАЛЬНОМ ВРЕМЕНИ (LIVE SCRAPING) ===
def scrape_website_content(url: str) -> str:
    """Заходит на сайт СДА, читает текст и отдает ИИ для актуальности фактов"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=5)
        response.encoding = 'utf-8'
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Удаляем скрипты и стили из HTML
            for script in soup(["script", "style", "header", "footer", "nav"]):
                script.extract()
            text = soup.get_text(separator=' ', strip=True)
            return text[:3500] # Ограничиваем, чтобы не перегрузить токен-лимит
    except Exception as e:
        logger.error(f"Ошибка парсинга сайта {url}: {e}")
    return ""

# === 6. ЛОГИКА ПОИСКА И ОЧИСТКИ ТЕКСТОВ ===
def clean_document_name(filename: str) -> str:
    """Тотальная очистка технических имен файлов для превращения их в красивые ссылки"""
    name = urllib.parse.unquote(filename)
    # Убираем расширения
    name = re.sub(r'\.(docx?|pdf|txt)$', '', name, flags=re.IGNORECASE)
    # Убираем все виды подчеркиваний и тире, делаем пробелы
    name = name.replace('_', ' ').replace('-', ' ')
    # Разлепляем слипшиеся слова (ПоложениеО -> Положение О)
    name = re.sub(r'([а-яёa-z])([А-ЯЁA-Z])', r'\1 \2', name)
    # Убираем приставки
    name = re.sub(r'^[\d\s]*СДА\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'^[\d\.\s]+', '', name)
    # Убираем мусор из названий (ДОПОЛНЕНО, Журнал и т.д.)
    name = re.sub(r'ДОПОЛНЕНО.*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'Журнал.*', '', name, flags=re.IGNORECASE)
    # Возвращаем даты к нормальному виду
    name = re.sub(r'(?<!от\s)(\d{2})\s(\d{2})\s(\d{4})', r'от \1.\2.\3', name)
    # Финальная зачистка пробелов
    name = re.sub(r'\s+', ' ', name).strip()
    return name.capitalize() if name else "Нормативный документ СДА"

def find_link_in_index(query: str) -> list:
    """Умный маршрутизатор по ключевым разделам сайта sdamp.ru"""
    q = query.lower()
    res = []
    
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
    """Итеративный RAG-поиск в 3 этапа"""
    if not db: return [], set()
    found_docs, sources_set =[], set()
    
    # 1. Прямой смысловой поиск
    docs_s1 = db.similarity_search(query, k=12)
    for d in docs_s1:
        s = clean_document_name(os.path.basename(d.metadata.get('source', 'Неизвестный')))
        sources_set.add(s)
        found_docs.append({'source': s, 'content': d.page_content, 'stage': 1})
    
    # 2. Поиск по ключевым словам
    for term in extract_keywords(query)[:3]:
        for d in db.similarity_search(term, k=5):
            s = clean_document_name(os.path.basename(d.metadata.get('source', 'Неизвестный')))
            if s not in [doc['source'] for doc in found_docs]:
                sources_set.add(s)
                found_docs.append({'source': s, 'content': d.page_content, 'stage': 2})
    
    # 3. Поиск связанных документов (ссылки внутри текста)
    for doc_ref in extract_document_references(found_docs)[:5]:
        for d in db.similarity_search(doc_ref, k=4):
            s = clean_document_name(os.path.basename(d.metadata.get('source', 'Неизвестный')))
            if s not in [doc['source'] for doc in found_docs]:
                sources_set.add(s)
                found_docs.append({'source': s, 'content': d.page_content, 'stage': 3})
                
    return found_docs, sources_set

def find_template(user_query: str) -> str | None:
    """Поиск файлов Word (шаблонов заявлений) для скачивания"""
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
    """Вытаскивает Уточняющие вопросы из ответа ИИ для создания кнопок"""
    suggestions =[]
    match = re.search(r'(🎯\s*УТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n?)(.+)', answer, re.IGNORECASE | re.DOTALL)
    if match:
        questions = re.findall(r'\[([^\]]+)\]|\b([А-Яа-яёЁ].*?\?)', match.group(2).strip())
        for q in questions:
            for part in q:
                if part.strip(): suggestions.append(part.strip())
    return suggestions[:3]

def clean_answer(answer: str) -> str:
    """Удаляет текстовый блок вопросов из ответа (так как они становятся кнопками)"""
    return re.sub(r'\n*🎯\s*УТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n?.*', '', answer, flags=re.DOTALL | re.IGNORECASE).strip()


# === 7. ГЛУБОКИЙ СИСТЕМНЫЙ ПРОМПТ И YANDEX GPT ===
DEEP_SYSTEM_PROMPT = """
Ты — Интеллектуальный юридический ассистент Сретенской духовной академии (СДА). 
Твоя роль — вежливый, эрудированный и внимательный эксперт-методист. Ты в совершенстве знаешь локальную нормативную базу Академии и ФЗ-273 "Об образовании".

ТВОЯ ЗАДАЧА: Давать исчерпывающие, юридически грамотные ответы, опираясь ТОЛЬКО на предоставленные фрагменты документов и актуальные данные с сайта.

ПРАВИЛА ФОРМИРОВАНИЯ ОТВЕТА (КРИТИЧЕСКИ ВАЖНО):

1. СТРУКТУРА ОТВЕТА (Соблюдать жестко):
📌 **ЗАКЛЮЧЕНИЕ** (1-2 предложения — прямой ответ на вопрос пользователя: разрешено, запрещено, возможно при определенных условиях).
📖 **ПРАВОВОЕ ОБОСНОВАНИЕ** (Детальный разбор ситуации. ОБЯЗАТЕЛЬНО ссылайся на конкретные пункты и статьи предоставленных документов. Пример: "На основании п. 5.1 Положения о текущем контроле..."). 
📋 **ПОРЯДОК ДЕЙСТВИЙ** (Если применимо — напиши пошаговый алгоритм действий для студента).
📎 **ДОКУМЕНТЫ** (Краткий перечень актов, которые регулируют данный вопрос).

2. АНАЛИЗ И КРИТИЧЕСКОЕ МЫШЛЕНИЕ:
- Внимательно читай КОНТЕКСТ. Если в локальных актах нет прямого ответа, но есть информация из блока "АКТУАЛЬНАЯ ИНФОРМАЦИЯ С САЙТА", используй её.
- Если ответа нет нигде — НЕ ВЫДУМЫВАЙ. Честно напиши: "К сожалению, в доступной мне нормативной базе СДА нет точного ответа на этот вопрос. Рекомендую обратиться лично в Учебную часть".

3. УТОЧНЯЮЩИЕ ВОПРОСЫ (ЭТО ОЧЕНЬ ВАЖНО):
В самом конце своего ответа ты ОБЯЗАН предложить 2-3 вопроса. 
ОНИ ДОЛЖНЫ БЫТЬ СФОРМУЛИРОВАНЫ ОТ ЛИЦА СТУДЕНТА (пользователя), как будто он хочет задать тебе следующий вопрос, чтобы углубиться в тему.
НЕ ЗАДАВАЙ вопросы пользователю (не пиши "Вам выслать бланк?"). ПРЕДЛАГАЙ ему готовые варианты!
Формат вывода строго: 
🎯 УТОЧНЯЮЩИЕ ВОПРОСЫ: [Вопрос 1?][Вопрос 2?]
"""

def call_yandex_gpt(history, current_question, context, site_links, web_context):
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {"Authorization": f"Api-Key {YANDEX_API_KEY}", "x-folder-id": FOLDER_ID}
    
    # Сборка памяти диалога (передаем системный промпт и историю)
    messages =[{"role": "system", "text": DEEP_SYSTEM_PROMPT}]
    for msg in history[-6:]: # Помним последние 6 реплик для хорошего контекста
        role = "assistant" if msg["role"] == "assistant" else "user"
        messages.append({"role": role, "text": msg["content"]})
    
    # Формируем тело финального запроса с учетом RAG и Парсинга
    user_payload = f"ВЫДЕРЖКИ ИЗ НОРМАТИВНЫХ АКТОВ:\n{context}\n"
    if web_context:
        user_payload += f"\n{web_context}\n"
        
    user_payload += f"\nВОПРОС ПОЛЬЗОВАТЕЛЯ: {current_question}\n\nДай ответ строго по структуре, с обоснованием и утоняющими вопросами от лица студента в конце."
    messages.append({"role": "user", "text": user_payload})

    payload = {
        "modelUri": f"gpt://{FOLDER_ID}/yandexgpt/latest",
        "completionOptions": {"temperature": 0.2, "maxTokens": "2000"},
        "messages": messages
    }
    
    res = requests.post(url, headers=headers, json=payload)
    if res.status_code == 200: 
        return res.json()['result']['alternatives'][0]['message']['text']
    else: 
        raise Exception(f"Ошибка YandexGPT: {res.text}")

def get_rag_response(question: str, chat_history: list):
    # 1. Поиск по базе (FAISS)
    docs, sources = iterative_search(question)
    
    # 2. Поиск по маршрутизатору сайта
    site_links = find_link_in_index(question)
    
    # 3. LIVE WEB SCRAPING (Парсинг сайта в реальном времени для критичных вопросов)
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
    
    try:
        # Вызов ИИ
        raw_answer = call_yandex_gpt(chat_history, question, context, site_links, live_web_context)
        
        # Формирование красивого блока источников с Markdown-гиперссылками на сайт
        clean_sources =[]
        for s in sources:
            if s.strip():
                clean_sources.append(f"📄 [{s}](https://sdamp.ru/sveden/document/)")
        
        sources_text = "\n\n".join(clean_sources)
        
        if site_links:
            links_text = "\n".join([f"🔗 [{link['title']}]({link['url']})" for link in site_links])
            sources_text += f"\n\n🌐 **Полезные разделы сайта:**\n{links_text}"
            
        suggestions = parse_suggestions(raw_answer)
        answer = clean_answer(raw_answer)
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
                st.download_button("📥 Скачать документ", f.read(), file_name=os.path.basename(msg["template"]), key=f"dl_{i}_{uuid.uuid4()}")
                
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
                    st.download_button("📥 Скачать документ", f.read(), file_name=fname, key=f"dl_new_{uuid.uuid4()}")
                
                st.session_state.messages.append({"role": "assistant", "content": ans, "template": t_path})
                save_message(st.session_state.user_id, prompt, ans)
                st.rerun()
                
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
                save_message(st.session_state.user_id, prompt, answer)
                st.rerun()
