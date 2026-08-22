# dynamo/clause-extract-api

Fine-tune + serve task repo, structured per the Dynamo authoring workflow.

```
proposal.md                          the 4-part proposal
instruction.md                       agent-facing prompt (Harbor)
task.toml                            Harbor manifest
environment/Dockerfile               shared image: deps + baked base model + train data
environment/data/train_clauses.jsonl synthetic training set (300 records, seed=42)
environment/data/generate_dataset.py deterministic generator used for both splits
solution/solve.sh, solve.py          oracle entrypoint (Harbor mounts this at /solution)
solution/train.py                    oracle train.py (copied to /app/train.py)
solution/serve.py                    oracle serve.py (copied to /app/serve.py)
solution/schemas.py                  oracle schemas.py (copied to /app/schemas.py)
solution/agent_solution.sh           oracle solution.sh (copied to /app/solution.sh)
solution/agent_instructions.md       oracle instructions.md (copied to /app/instructions.md)
tests/test.sh                        verifier entrypoint -> /logs/verifier/reward.txt
tests/test_outputs.py                one function per success criterion
tests/eval_clauses.jsonl             held-out set: seed=1337, value offset +1000
                                      (disjoint from training seed=42, offset 0)
```

## Status / what's been checked here

Everything above was authored and structurally validated in this sandbox:

- All Python files (`train.py`, `serve.py`, `schemas.py`, `solve.py`, `test_outputs.py`)
  pass `py_compile` (syntactically valid).
- `task.toml` parses cleanly with `tomllib`.
- The synthetic dataset generator is deterministic (fixed seeds) and was actually run
  to produce both `environment/data/train_clauses.jsonl` (300 records) and
  `tests/eval_clauses.jsonl` (50 records, disjoint seed + numeric-value range).
- `instruction.md` is ~1,060 estimated tokens, under the 1,500-token cap, and ends with
  the exact required timeout line.

**What has *not* been run in this sandbox:** the actual `harbor run -p . --agent oracle`
end-to-end pass. That requires installing `torch`/`transformers`/`peft`/`fastapi`
(network access) and building the Docker image (which bakes `distilgpt2` in at build
time), neither of which this sandbox has available. Before merging, run:

```bash
harbor run -p . --agent oracle   # expect reward 1.0
harbor run -p . --agent nop      # expect reward < 1.0
```

If the oracle's `clause_type` accuracy on `tests/eval_clauses.jsonl` doesn't
comfortably clear the 0.70 band in practice, the most likely knobs are
`train.py`'s `num_train_epochs` / `learning_rate` and the LoRA `target_modules` list
(GPT-2-family attention module names can shift between `transformers` versions —
confirm `["c_attn"]` matches whatever `distilgpt2`'s attention layer is actually named
in the pinned version before treating a training-run failure as a task-design problem).

## Known placeholders to fix against the real repo

- `task.toml`'s `category`/`subcategory` are left blank (per the workflow doc, these
  are pre-seeded by the Dynamo team when the task repo is created).
- `task_objective` / `artifact_type` labels are best-guess snake_case values matching
  the shape described in the workflow doc; cross-check them against the actual
  `.dynamo/diversity-taxonomy.toml` in the real repo, which wasn't available here.
# clause-extract-api
