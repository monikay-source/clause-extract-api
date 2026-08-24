#!/usr/bin/env python3
"""
Oracle solution. Assembles the required deliverables under /app and runs the
training stage so /app/model/adapter/ exists, exactly as a correct agent
submission should look at the end of the episode. Does NOT leave the API
server running — the verifier starts serve.py independently (it does not
trust or execute /app/solution.sh).
"""
import os
import shutil
import subprocess
import sys

SOLUTION_DIR = "/solution"
APP_DIR = "/app"

FILE_MAP = {
    "schemas.py": "schemas.py",
    "train.py": "train.py",
    "serve.py": "serve.py",
    "agent_solution.sh": "solution.sh",
    "agent_instructions.md": "instructions.md",
}


def main():
    os.makedirs(APP_DIR, exist_ok=True)
    os.makedirs(os.path.join(APP_DIR, "logs"), exist_ok=True)

    for src_name, dst_name in FILE_MAP.items():
        src = os.path.join(SOLUTION_DIR, src_name)
        dst = os.path.join(APP_DIR, dst_name)
        shutil.copyfile(src, dst)
        print(f"wrote {dst}")

    os.chmod(os.path.join(APP_DIR, "solution.sh"), 0o755)

    print("Running training...")
    result = subprocess.run(
        [sys.executable, os.path.join(APP_DIR, "train.py")],
        cwd=APP_DIR,
    )
    if result.returncode != 0:
        print("train.py failed", file=sys.stderr)
        sys.exit(1)

    print("Oracle solution complete.")


if __name__ == "__main__":
    main()
