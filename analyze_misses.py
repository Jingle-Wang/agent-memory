import sqlite3, json

conn = sqlite3.connect('runs/benchmarks/phase1-v1/memory.db')
cursor = conn.cursor()

with open('runs/benchmarks/phase1-v1/scores.jsonl') as f:
    scores = [json.loads(line) for line in f]

with open('data/benchmarks/locomo/locomo10-conv26.json') as f:
    qa_data = json.load(f)[0]['qa']

miss_ids = [0, 1, 2, 3, 4, 7, 13, 18]

gold_evidence = {
    0: ['conv-26::D1:3'],
    1: ['conv-26::D1:12'],
    2: ['conv-26::D1:9', 'conv-26::D1:11'],
    3: ['conv-26::D2:8'],
    4: ['conv-26::D1:5'],
    7: ['conv-26::D3:13', 'conv-26::D2:14'],
    13: ['conv-26::D4:13', 'conv-26::D1:11'],
    18: ['conv-26::D6:16', 'conv-26::D4:6', 'conv-26::D8:32'],
}

for qid in miss_ids:
    q = qa_data[qid]
    score = scores[qid]
    
    print(f'\n{"="*70}')
    print(f'q:{qid} - {q["question"][:90]}')
    print(f'Gold answer: {q["answer"]}')
    print(f'Gold evidence src: {gold_evidence[qid]}')
    print(f'Recall@10: {score["retrieval"]["recall_at_10"]}')
    
    gold_mem_ids = set()
    for src in gold_evidence[qid]:
        cursor.execute("SELECT id FROM memories WHERE source_event_id = ?", (src,))
        for row in cursor.fetchall():
            gold_mem_ids.add(row[0])
    
    ret_mem_ids = set(score['retrieved_memory_ids'][:10])
    overlap = gold_mem_ids & ret_mem_ids
    
    if overlap:
        print(f'*** OVERLAP: {overlap} ***')
    else:
        print(f'*** NO OVERLAP - gold memories not in top 10 ***')
    
    print(f'Gold mem_ids ({len(gold_mem_ids)} total):')
    for gmid in list(gold_mem_ids)[:5]:
        cursor.execute("SELECT content, memory_type, importance, confidence FROM memories WHERE id = ?", (gmid,))
        row = cursor.fetchone()
        if row:
            c = row[0].replace('\n', ' ')
            print(f'  {gmid} [{row[1]}] imp={row[2]:.2f} conf={row[3]:.2f}: {c[:120]}')
    
    print(f'\nTop-5 Retrieved (what we got instead):')
    for i in range(min(5, len(score['retrieved_memory_ids']))):
        mid = score['retrieved_memory_ids'][i]
        sid = score['retrieved_source_event_ids'][i]
        cursor.execute("SELECT content, memory_type, importance, confidence FROM memories WHERE id = ?", (mid,))
        row = cursor.fetchone()
        if row:
            c = row[0].replace('\n', ' ')
            print(f'  [{i+1}] {mid} src={sid} [{row[1]}] imp={row[2]:.2f}: {c[:130]}')

conn.close()
