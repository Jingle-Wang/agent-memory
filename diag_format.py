#!/usr/bin/env python3
"""Diagnose source_event_id vs evidence_turn_id format matching.
Simulates the exact Rust scoping logic from src/benchmark/loader.rs.
"""
import json

def scoped_id(scope: str, id_: str) -> str:
    """Mirrors Rust: format!("{scope}::{id}")"""
    return f"{scope}::{id_}"

def main():
    with open("data/benchmarks/locomo/locomo10.json") as f:
        data = json.load(f)

    total_questions = 0
    total_evidence_missing = 0
    total_evidence_matched = 0
    missing_examples = []

    for record_index, record in enumerate(data):
        sample_id = record.get("sample_id", f"locomo-{record_index}")
        conv = record.get("conversation", {})

        # --- Build the set of turn IDs exactly as Rust would scope them ---
        # Parse sessions (mirrors Rust loader logic)
        session_numbers = []
        for key in conv:
            if key.startswith("session_"):
                num_part = key[len("session_"):]
                try:
                    num = int(num_part)
                    session_numbers.append((num, key))
                except ValueError:
                    pass  # skip _date_time keys

        session_numbers.sort(key=lambda x: x[0])

        turn_ids = set()
        turn_id_to_info = {}
        for session_num, key in session_numbers:
            turns = conv.get(key, [])
            if not isinstance(turns, list):
                continue
            timestamp_key = f"session_{session_num}_date_time"
            timestamp = conv.get(timestamp_key, None)

            for turn_index, turn in enumerate(turns):
                dia_id = turn.get("dia_id") or turn.get("id")
                if dia_id:
                    tid = scoped_id(sample_id, dia_id)
                else:
                    tid = scoped_id(sample_id, f"s{session_num}:t{turn_index}")

                turn_ids.add(tid)
                turn_id_to_info[tid] = {
                    "dia_id": dia_id,
                    "session": session_num,
                    "text": turn.get("text", "")[:80],
                }

        # --- Build evidence turn IDs exactly as Rust would scope them ---
        qa = record.get("qa", [])
        for qi, question in enumerate(qa):
            total_questions += 1
            evidence = question.get("evidence", [])
            if not evidence:
                continue

            for ev_id in evidence:
                ev_scoped = scoped_id(sample_id, ev_id)
                if ev_scoped in turn_ids:
                    total_evidence_matched += 1
                else:
                    total_evidence_missing += 1
                    if len(missing_examples) < 10:
                        missing_examples.append({
                            "record": record_index,
                            "question": qi,
                            "sample_id": sample_id,
                            "evidence_raw": ev_id,
                            "evidence_scoped": ev_scoped,
                            "question_text": question.get("question", "")[:100],
                        })

    print(f"=== Format Diagnosis ===")
    print(f"Total records: {len(data)}")
    print(f"Total questions: {total_questions}")
    print(f"Evidence IDs that MATCH a turn ID: {total_evidence_matched}")
    print(f"Evidence IDs that DON'T match any turn ID: {total_evidence_missing}")

    if missing_examples:
        print(f"\n=== Missing Evidence Examples (first 10) ===")
        for ex in missing_examples:
            print(f"  Record {ex['record']} Q{ex['question']}: evidence={ex['evidence_raw']} → scoped={ex['evidence_scoped']}")
            print(f"    Question: {ex['question_text']}")

            # Show what turn IDs exist with similar format
            conv = data[ex['record']].get("conversation", {})
            # Find the session this evidence might refer to
            ev_parts = ex['evidence_raw'].split(":")
            if len(ev_parts) == 2:
                dia_prefix = ev_parts[0]  # e.g., "D1"
                sess_key = f"session_{dia_prefix[1:]}"  # e.g., "session_1"
                if sess_key in conv:
                    turns = conv[sess_key]
                    turn_dia_ids = [t.get("dia_id", "?") for t in turns[:5]]
                    print(f"    Session {sess_key} has {len(turns)} turns, dia_ids (first 5): {turn_dia_ids}")
                    print(f"    Evidence {ex['evidence_raw']} refers to turn #{ev_parts[1]} in this session")
                    if int(ev_parts[1]) < len(turns):
                        target = turns[int(ev_parts[1]) - 1]
                        print(f"    Target turn dia_id={target.get('dia_id')}, text={target.get('text','')[:80]}")
                else:
                    print(f"    Session {sess_key} NOT FOUND in conversation keys")

    # --- Show a few example matches ---
    print(f"\n=== Example Matched IDs ===")
    rec0 = data[0]
    sid0 = rec0["sample_id"]
    conv0 = rec0["conversation"]
    # First session, first 3 turns
    turns_s1 = conv0.get("session_1", [])
    for t in turns_s1[:3]:
        raw_dia = t.get("dia_id", "?")
        scoped = scoped_id(sid0, raw_dia)
        print(f"  dia_id={raw_dia} → turn.id={scoped}")

    # First question evidence
    qa0 = rec0["qa"]
    if qa0:
        q0 = qa0[0]
        ev = q0.get("evidence", [])
        for e in ev:
            print(f"  evidence={e} → evidence_turn_id={scoped_id(sid0, e)}")

    print(f"\n=== Verdict ===")
    if total_evidence_missing == 0:
        print("ALL evidence_turn_ids match existing turn IDs. Format is consistent.")
        print("The 0% recall is NOT caused by ID format mismatch in scoping.")
        print("Root cause must be in the retrieval/ranking system.")
    else:
        print(f"{total_evidence_missing} evidence IDs have NO matching turn — format mismatch confirmed!")
        print("This is the root cause of 0% recall.")


if __name__ == "__main__":
    main()
