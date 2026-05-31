# -*- coding: utf-8 -*-
"""
Сквозная оценка (A2): полный конвейер retrieval -> YandexGPT.
Прогоняет тест-сет через реальную систему, сохраняет ответы для оценки.

Авто-метрики:
  - in_corpus: процитирован ли релевантный документ (gold-токен в тексте ответа),
               соблюдена ли структура ответа (наличие блоков 📌 и 📖)
  - граничные: доля корректных отказов (фраза «не найден / обратитесь в Учебную часть»)
Ручная оценка (заполняется в answers_for_grading.csv):
  - ans_correct (0/1), cite_correct (0/1), hallucination (0/1)

Ключ НЕ хранится в коде. Запуск:
  YANDEX_API_KEY=<секрет> python e2e_eval.py      (Linux/Mac)
  $env:YANDEX_API_KEY="<секрет>"; python e2e_eval.py   (PowerShell)
"""
import sys, os, re, csv, json, time
import requests

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def _find_repo():
    here = os.path.dirname(os.path.abspath(__file__))
    # eval/ внутри репозитория: ..; либо drafts/eval/: ../../super_new_RAG
    for cand in (os.path.join(here, '..'),
                 os.path.join(here, '..', '..', 'super_new_RAG')):
        if os.path.isdir(os.path.normpath(os.path.join(cand, 'sretensk_db'))):
            return os.path.normpath(cand)
    return os.path.normpath(os.path.join(here, '..'))
REPO = _find_repo()
DB_PATH = os.path.join(REPO, 'sretensk_db')  # реальный FAISS-индекс
SITE_INDEX_FILE = os.path.join(REPO, 'docs', 'site_index.json')
HERE = os.path.dirname(os.path.abspath(__file__))
QUESTIONS = os.path.join(HERE, 'test_questions.csv')
OUT = os.path.join(HERE, 'answers_for_grading.csv')

YANDEX_API_KEY = os.getenv('YANDEX_API_KEY', '')
FOLDER_ID = os.getenv('FOLDER_ID', 'b1g6jhk9eapudn6lom6c')
if not YANDEX_API_KEY:
    sys.exit('Не задан YANDEX_API_KEY (переменная окружения).')

print('Загрузка эмбеддингов и индекса...')
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
site_index = {'pages': [], 'documents': []}
if os.path.exists(SITE_INDEX_FILE):
    with open(SITE_INDEX_FILE, encoding='utf-8') as f:
        site_index = json.load(f)

DEEP_SYSTEM_PROMPT = """
Ты — Интеллектуальный юридический ассистент Сретенской духовной академии (СДА).
Твоя роль — вежливый, эрудированный и внимательный эксперт-методист. Ты в совершенстве знаешь локальную нормативную базу Академии и ФЗ-273 "Об образовании".
ТВОЯ ЗАДАЧА: Давать исчерпывающие, юридически грамотные ответы, опираясь ТОЛЬКО на предоставленные фрагменты документов и актуальные данные с сайта.
ПРАВИЛА ФОРМИРОВАНИЯ ОТВЕТА (КРИТИЧЕСКИ ВАЖНО):
1. СТРУКТУРА ОТВЕТА (Соблюдать жестко):
📌 **ЗАКЛЮЧЕНИЕ** (1-2 предложения — суть ответа).
📖 **ПРАВОВОЕ ОБОСНОВАНИЕ** (Детальный разбор. ОБЯЗАТЕЛЬНО ссылайся на конкретные пункты и статьи предоставленных документов).
📋 **ПОРЯДОК ДЕЙСТВИЙ** (пошаговый алгоритм для студента).
📎 **ДОКУМЕНТЫ** (перечень использованных актов).
2. Если ответа нет нигде — НЕ ВЫДУМЫВАЙ. Честно напиши: "К сожалению, в доступной мне нормативной базе СДА нет точного ответа на этот вопрос. Рекомендую обратиться лично в Учебную часть".
3. В конце предложи 2-3 УТОЧНЯЮЩИХ ВОПРОСА от лица студента.
"""


def extract_keywords(query):
    stop_words = {'как', 'что', 'где', 'когда', 'почему', 'можно', 'нужно', 'могу', 'ли', 'или', 'подскаж'}
    words = re.findall(r'\b[а-яёА-ЯЁ]{4,}\b', query.lower())
    return [w for w in words if w not in stop_words]


def extract_document_references(docs):
    refs = []
    patterns = [r'[Пп]оложение[а-яё\s]*["«]([^"]+)["»]', r'[Пп]риказ[а-яё\s]*№?\s*\d+.*["«]([^"]+)["»]']
    for doc in docs:
        for p in patterns:
            refs.extend(re.findall(p, doc['content']))
    return list(set(refs))[:10]


def clean_name(filename):
    name = re.sub(r'\.(docx?|pdf|txt)$', '', filename, flags=re.IGNORECASE)
    name = name.replace('_', ' ').replace('-', ' ')
    name = re.sub(r'^[\d\.\s]+', '', name)
    return re.sub(r'\s+', ' ', name).strip()


def iterative_search(query):
    found = []
    for d in db.similarity_search(query, k=12):
        found.append({'source': clean_name(os.path.basename(d.metadata.get('source', '?'))),
                      'raw': os.path.basename(d.metadata.get('source', '?')).lower(),
                      'content': d.page_content, 'stage': 1})
    existing = {x['source'] for x in found}
    for term in extract_keywords(query)[:3]:
        for d in db.similarity_search(term, k=5):
            s = clean_name(os.path.basename(d.metadata.get('source', '?')))
            if s not in existing:
                existing.add(s)
                found.append({'source': s, 'raw': os.path.basename(d.metadata.get('source', '?')).lower(),
                              'content': d.page_content, 'stage': 2})
    for ref in extract_document_references(found)[:5]:
        for d in db.similarity_search(ref, k=4):
            s = clean_name(os.path.basename(d.metadata.get('source', '?')))
            if s not in existing:
                existing.add(s)
                found.append({'source': s, 'raw': os.path.basename(d.metadata.get('source', '?')).lower(),
                              'content': d.page_content, 'stage': 3})
    return found


def call_yandex(question, context):
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {"Authorization": f"Api-Key {YANDEX_API_KEY}", "x-folder-id": FOLDER_ID}
    messages = [{"role": "system", "text": DEEP_SYSTEM_PROMPT},
                {"role": "user", "text": f"КОНТЕКСТ ИЗ ДОКУМЕНТОВ:\n{context}\n\nВОПРОС ПОЛЬЗОВАТЕЛЯ: {question}"}]
    payload = {"modelUri": f"gpt://{FOLDER_ID}/yandexgpt/latest",
               "completionOptions": {"temperature": 0.2, "maxTokens": "2000"}, "messages": messages}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code == 200:
        return r.json()['result']['alternatives'][0]['message']['text']
    raise Exception(f"YandexGPT {r.status_code}: {r.text[:200]}")


REFUSAL = ['не найден', 'обратит', 'учебную часть', 'нет точного ответа', 'нет ответа']

with open(QUESTIONS, encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

out_rows = []
for r in rows:
    gold = [t.strip().lower() for t in r['gold_tokens'].split('|') if t.strip()]
    found = sorted(iterative_search(r['question']), key=lambda x: x['stage'])[:15]
    context = "\n\n".join(f"--- ФРАГМЕНТ (Источник: {d['source']}) ---\n{d['content']}" for d in found)
    try:
        ans = call_yandex(r['question'], context)
    except Exception as e:
        ans = f"[ОШИБКА: {e}]"
    low = ans.lower()
    refused = int(any(p in low for p in REFUSAL))
    cite_gold_auto = int(any(tok in low for tok in gold)) if gold else ''
    structure_ok = int('📌' in ans and '📖' in ans)
    out_rows.append({
        'id': r['id'], 'theme': r['theme'], 'qtype': r['qtype'],
        'in_corpus': r['in_corpus'], 'question': r['question'],
        'refused_auto': refused, 'cite_gold_auto': cite_gold_auto, 'structure_ok': structure_ok,
        'answer': ans.replace('\n', ' \\n '),
        'ans_correct': '', 'cite_correct': '', 'hallucination': '',
    })
    print(f"#{r['id']:>2} done (refused={refused}, struct={structure_ok})")
    time.sleep(0.3)

with open(OUT, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['id', 'theme', 'qtype', 'in_corpus', 'question',
                                      'refused_auto', 'cite_gold_auto', 'structure_ok', 'answer',
                                      'ans_correct', 'cite_correct', 'hallucination'])
    w.writeheader()
    w.writerows(out_rows)

# Авто-сводка
inc = [r for r in out_rows if r['in_corpus'] == '1']
edge = [r for r in out_rows if r['in_corpus'] == '0']
print('\n' + '=' * 60)
print(f"Цитирование gold (авто) по корпусу: {sum(int(r['cite_gold_auto'] or 0) for r in inc)}/{len(inc)}")
print(f"Структура ответа соблюдена: {sum(r['structure_ok'] for r in out_rows)}/{len(out_rows)}")
print(f"Корректный отказ на граничных: {sum(r['refused_auto'] for r in edge)}/{len(edge)}")
print(f"\nОтветы для ручной оценки: {OUT}")
