import sqlite3, json, re

conn = sqlite3.connect('runs/benchmarks/phase1-v1/memory.db')
cursor = conn.cursor()

with open('runs/benchmarks/phase1-v1/scores.jsonl') as f:
    scores = [json.loads(line) for line in f]
with open('data/benchmarks/locomo/locomo10-conv26.json') as f:
    qa_data = json.load(f)[0]['qa']

miss_ids = [0, 1, 2, 3, 4, 7, 13, 18]
gold_evidence = {
    0: ['conv-26::D1:3'], 1: ['conv-26::D1:12'], 
    2: ['conv-26::D1:9', 'conv-26::D1:11'], 3: ['conv-26::D2:8'],
    4: ['conv-26::D1:5'], 7: ['conv-26::D3:13', 'conv-26::D2:14'],
    13: ['conv-26::D4:13', 'conv-26::D1:11'],
    18: ['conv-26::D6:16', 'conv-26::D4:6', 'conv-26::D8:32'],
}

def simple_tokenize(s):
    return set(re.findall(r'[a-z0-9]+', s.lower()))

def jaccard(qt, mt):
    if not qt or not mt: return 0.0
    return len(qt & mt) / len(qt | mt)

for qid in miss_ids:
    q = qa_data[qid]
    qtokens = simple_tokenize(q['question'])
    score = scores[qid]
    
    print(f'\n{"="*70}')
    print(f'q:{qid} - {q["question"][:90]}')
    print(f'Query tokens ({len(qtokens)}): {sorted(qtokens)}')
    print(f'Gold evidence: {gold_evidence[qid]}')
    
    gold_mem_ids = set()
    for src in gold_evidence[qid]:
        cursor.execute("SELECT id FROM memories WHERE source_event_id = ?", (src,))
        for row in cursor.fetchall():
            gold_mem_ids.add(row[0])
    
    print(f'Gold memories:')
    for gmid in list(gold_mem_ids)[:4]:
        cursor.execute("SELECT content FROM memories WHERE id = ?", (gmid,))
        row = cursor.fetchone()
        if row:
            c = row[0].replace('\n', ' ')
            mt = simple_tokenize(c)
            j = jaccard(qtokens, mt)
            common = qtokens & mt
            print(f'  J={j:.3f} common({len(common)}): {sorted(common)}')
            print(f'  >> {c[:120]}')
    
    print(f'Top-5 Retrieved:')
    for i in range(min(5, len(score['retrieved_memory_ids']))):
        mid = score['retrieved_memory_ids'][i]
        cursor.execute("SELECT content FROM memories WHERE id = ?", (mid,))
        row = cursor.fetchone()
        if row:
            c = row[0].replace('\n', ' ')
            mt = simple_tokenize(c)
            j = jaccard(qtokens, mt)
            common = qtokens & mt
            print(f'  [{i+1}] J={j:.3f} common({len(common)}): {sorted(common)}')
            print(f'  >> {c[:130]}')

conn.close()
