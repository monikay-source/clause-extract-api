#!/usr/bin/env python3
"""
Deterministic synthetic contract-clause generator.

Produces two disjoint sets:
  - train_clauses.jsonl  (baked into environment/, agent may use for fine-tuning)
  - the held-out eval set is generated separately by tests/generate_eval.py
    using a different seed/value-range so it is NOT visible to the agent.

Every record: {"text": "...", "clause_type": "...", "extracted_value": "..."}

Five clause types, each built from a template family with randomized
boilerplate (party names, section numbers, filler sentences) so the
task requires actual pattern learning rather than pure string lookup,
while remaining tractable for a small LoRA-tuned model.
"""
import json
import random
import sys

CLAUSE_TYPES = ["termination", "payment_terms", "confidentiality", "renewal", "leave_benefits"]

PARTIES = ["Acme Corp", "Beacon Industries", "Crestline LLC", "Dunmore Partners",
           "Everline Systems", "Fjord Analytics", "Granite Bay Co", "Halden Group"]

FILLERS = [
    "This clause is part of the broader service agreement executed by the parties.",
    "Nothing in this section shall be construed to limit any other rights herein.",
    "The parties acknowledge that this provision is material to the Agreement.",
    "All notices under this section shall be delivered in writing.",
    "This provision survives expiration of the Agreement where applicable.",
]

TEMPLATES = {
    "termination": (
        lambda n, a, b, filler, sec:
        f"Section {sec}. Termination. This Agreement may be terminated by either "
        f"{a} or {b} upon {n} days written notice to the other party. {filler}",
        lambda n: f"{n} days",
    ),
    "payment_terms": (
        lambda n, a, b, filler, sec:
        f"Section {sec}. Payment Terms. {a} shall remit payment to {b} within "
        f"{n} days of receipt of a valid invoice. {filler}",
        lambda n: f"{n} days",
    ),
    "confidentiality": (
        lambda n, a, b, filler, sec:
        f"Section {sec}. Confidentiality. The Receiving Party shall maintain the "
        f"confidentiality of all Confidential Information disclosed by {a} to {b} "
        f"for a period of {n} years following the date of disclosure. {filler}",
        lambda n: f"{n} years",
    ),
    "renewal": (
        lambda n, a, b, filler, sec:
        f"Section {sec}. Renewal. This Agreement between {a} and {b} shall "
        f"automatically renew for successive periods of {n} months unless either "
        f"party provides notice of non-renewal. {filler}",
        lambda n: f"{n} months",
    ),
    "leave_benefits": (
        lambda n, a, b, filler, sec:
        f"Section {sec}. Leave Benefits. Each employee of {a} shall be entitled "
        f"to {n} days of paid leave per calendar year, subject to the policies of {b}. {filler}",
        lambda n: f"{n} days",
    ),
}

VALUE_RANGES = {
    "termination": (5, 60),
    "payment_terms": (10, 90),
    "confidentiality": (1, 10),
    "renewal": (1, 24),
    "leave_benefits": (5, 30),
}


def generate(seed: int, per_class: int, value_range_offset: int = 0):
    rng = random.Random(seed)
    records = []
    for clause_type in CLAUSE_TYPES:
        text_fn, value_fn = TEMPLATES[clause_type]
        lo, hi = VALUE_RANGES[clause_type]
        lo += value_range_offset
        hi += value_range_offset
        for i in range(per_class):
            n = rng.randint(lo, hi)
            a, b = rng.sample(PARTIES, 2)
            filler = rng.choice(FILLERS)
            sec = rng.randint(1, 20)
            text = text_fn(n, a, b, filler, sec)
            records.append({
                "text": text,
                "clause_type": clause_type,
                "extracted_value": value_fn(n),
            })
    rng.shuffle(records)
    return records


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "train_clauses.jsonl"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    per_class = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    offset = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    recs = generate(seed, per_class, offset)
    with open(out_path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(recs)} records to {out_path}")
