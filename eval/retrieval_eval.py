# -*- coding: utf-8 -*-
"""
Офлайн-оценка качества ретривера трёхступенчатого RAG-конвейера СДА.
Воспроизводит iterative_search из app.py (super_new_RAG) и измеряет,
доходит ли релевантный документ до контекста языковой модели.

Метрики:
  - Recall@15 (контекст): попал ли gold-документ в 15 фрагментов, передаваемых LLM
  - Recall@12 (только семантика, ступень 1)
  - MRR: средний обратный ранг первого релевантного фрагмента
  - Вклад ступеней: на какой ступени впервые найден релевантный документ

Запуск:
  python retrieval_eval.py
"""
import sys, os, re, csv

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def _find_db():
    here = os.path.dirname(os.path.abspath(__file__))
    # eval/ внутри репозитория: ../sretensk_db; либо drafts/eval/: ../../super_new_RAG/sretensk_db
    for cand in (os.path.join(here, '..', 'sretensk_db'),
                 os.path.join(here, '..', '..', 'super_new_RAG', 'sretensk_db')):
        if os.path.isdir(os.path.normpath(cand)):
            return os.path.normpath(cand)
    return os.path.normpath(os.path.join(here, '..', 'sretensk_db'))
DB_PATH = _find_db()  # реальный FAISS-индекс
HERE = os.path.dirname(os.path.abspath(__file__))
QUESTIONS = os.path.join(HERE, 'test_questions.csv')
OUT = os.path.join(HERE, 'retrieval_results.csv')

CONTEXT_TOP = 15  # столько фрагментов get_rag_response передаёт в YandexGPT

print('Загрузка эмбеддингов (paraphrase-multilingual-MiniLM-L12-v2)...')
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
print('Загрузка FAISS-индекса...')
db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
print(f'Векторов в индексе: {db.index.ntotal}\n')


# ── Функции из app.py (воспроизведены дословно) ──────────────────────────────
def extract_keywords(query):
    stop_words = {'как', 'что', 'где', 'когда', 'почему', 'можно', 'нужно', 'могу', 'ли', 'или', 'подскаж'}
    words = re.findall(r'\b[а-яёА-ЯЁ]{4,}\b', query.lower())
    return [w for w in words if w not in stop_words]


def extract_document_references(docs):
    references = []
    patterns = [r'[Пп]оложение[а-яё\s]*["«]([^"]+)["»]', r'[Пп]риказ[а-яё\s]*№?\s*\d+.*["«]([^"]+)["»]']
    for doc in docs:
        for p in patterns:
            references.extend(re.findall(p, doc['content']))
    return list(set(references))[:10]


def raw_source(d):
    return os.path.basename(d.metadata.get('source', 'Неизвестный')).lower()


def iterative_search(query):
    """Возвращает found_docs со стадиями (как в app.py, но с сырым именем источника)."""
    found_docs = []
    for d in db.similarity_search(query, k=12):
        found_docs.append({'source': raw_source(d), 'content': d.page_content, 'stage': 1})
    existing = {doc['source'] for doc in found_docs}
    for term in extract_keywords(query)[:3]:
        for d in db.similarity_search(term, k=5):
            s = raw_source(d)
            if s not in existing:
                existing.add(s)
                found_docs.append({'source': s, 'content': d.page_content, 'stage': 2})
    for doc_ref in extract_document_references(found_docs)[:5]:
        for d in db.similarity_search(doc_ref, k=4):
            s = raw_source(d)
            if s not in existing:
                existing.add(s)
                found_docs.append({'source': s, 'content': d.page_content, 'stage': 3})
    return found_docs


def gold_hit(source, gold_tokens):
    return any(tok and tok in source for tok in gold_tokens)


# ── Прогон ───────────────────────────────────────────────────────────────────
rows = []
with open(QUESTIONS, encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

results = []
in_corpus = [r for r in rows if r['in_corpus'] == '1']
edge = [r for r in rows if r['in_corpus'] == '0']

hit15 = hit12 = 0
rr_sum = 0.0
stage_first = {1: 0, 2: 0, 3: 0, 'miss': 0}

for r in in_corpus:
    gold = [t.strip().lower() for t in r['gold_tokens'].split('|') if t.strip()]
    found = iterative_search(r['question'])
    found_sorted = sorted(found, key=lambda x: x['stage'])
    context = found_sorted[:CONTEXT_TOP]

    # стадия 1 (семантика) отдельно
    stage1_sources = [d['source'] for d in found if d['stage'] == 1]
    h12 = any(gold_hit(s, gold) for s in stage1_sources)

    # ранг первого релевантного фрагмента в контексте
    rank = 0
    for i, d in enumerate(context, 1):
        if gold_hit(d['source'], gold):
            rank = i
            break
    h15 = rank > 0

    # на какой стадии впервые найден релевантный источник
    first_stage = 'miss'
    for d in found_sorted:
        if gold_hit(d['source'], gold):
            first_stage = d['stage']
            break

    hit15 += int(h15)
    hit12 += int(h12)
    rr_sum += (1.0 / rank) if rank else 0.0
    stage_first[first_stage] += 1

    results.append({
        'id': r['id'], 'theme': r['theme'], 'qtype': r['qtype'],
        'hit@15': int(h15), 'hit@12_sem': int(h12),
        'rank': rank, 'first_stage': first_stage,
        'n_unique_sources': len({d['source'] for d in found}),
    })
    flag = 'OK ' if h15 else 'MISS'
    print(f"[{flag}] #{r['id']:>2} ({r['qtype']:8}) rank={rank or '-':>2} st={first_stage}  {r['question'][:54]}")

n = len(in_corpus)
print('\n' + '=' * 64)
print(f'Вопросов по корпусу: {n}')
print(f'Recall@15 (контекст LLM): {hit15}/{n} = {hit15/n:.1%}')
print(f'Recall@12 (только семантика, ступень 1): {hit12}/{n} = {hit12/n:.1%}')
print(f'MRR: {rr_sum/n:.3f}')
print(f'Впервые найдено на ступени: 1={stage_first[1]}, 2={stage_first[2]}, 3={stage_first[3]}, miss={stage_first["miss"]}')
print('=' * 64)

# Сохранение
with open(OUT, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['id', 'theme', 'qtype', 'hit@15', 'hit@12_sem', 'rank', 'first_stage', 'n_unique_sources'])
    w.writeheader()
    w.writerows(results)
print(f'\nДетальные результаты: {OUT}')
print(f'Граничных вопросов (вне корпуса) для e2e-теста: {len(edge)}')
