#!/usr/bin/env python3
"""Run the authors' upstream OptIForest demo and archive stdout/metadata."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, platform
from pathlib import Path
from datetime import datetime, timezone


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True, help="Path to locally cloned official OptIForest repo")
    p.add_argument("--dataset", default="ad")
    p.add_argument("--threshold", type=int, default=403)
    p.add_argument("--branch", type=int, default=0)
    p.add_argument("--python", default=sys.executable, help="Interpreter/environment that has upstream dependencies installed")
    p.add_argument("--log-dir", type=Path, default=Path("logs"))
    a = p.parse_args()
    repo = a.repo.resolve()
    script = repo / "demo.py"
    if not script.exists():
        raise SystemExit(f"Could not find {script}; --repo should point to the upstream OptIForest checkout.")
    a.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = a.log_dir / f"official_{a.dataset}_{stamp}.txt"
    metadata_path = a.log_dir / f"official_{a.dataset}_{stamp}.json"
    command = [a.python, str(script), f"--dataset={a.dataset}", f"--threshold={a.threshold}", f"--branch={a.branch}"]
    started = time.time()
    proc = subprocess.run(command, cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - started
    log_path.write_text(proc.stdout, encoding="utf-8")
    metadata = {
        "timestamp_utc": stamp,
        "command": command,
        "working_directory": str(repo),
        "python_executable": a.python,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "elapsed_seconds": round(elapsed, 4),
        "return_code": proc.returncode,
        "stdout_log": str(log_path),
        "dataset": a.dataset,
        "threshold": a.threshold,
        "branch": a.branch,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(proc.stdout)
    print(json.dumps(metadata, indent=2))
    if proc.returncode != 0:
        raise SystemExit(f"Upstream command failed (exit {proc.returncode}). Check {log_path} and verify dependencies/data format.")
    print(f"Saved raw output to {log_path}")
    print(f"Saved run metadata to {metadata_path}")

if __name__ == "__main__":
    main()
