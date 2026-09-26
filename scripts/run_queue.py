"""GPU 작업 큐: 명령을 순서대로 실행하고 outputs/logs/<name>.log 에 기록. 하나가 실패해도 나머지는 계속.

python scripts/run_queue.py queue.yaml
queue.yaml:
  - name: det
    cmd: python scripts/train_det.py --config exp_det
  - name: cls_mono
    cmd: python scripts/train_cls.py --config exp_cls_mono
상태: outputs/logs/queue_status.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    queue_file = Path(sys.argv[1])
    jobs = yaml.safe_load(queue_file.read_text(encoding="utf-8"))
    log_dir = ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / f"queue_status_{queue_file.stem}.json"
    status = {"queue": str(queue_file), "started": datetime.now().isoformat(), "jobs": []}
    for job in jobs:
        name, cmd = job["name"], job["cmd"]
        log = log_dir / f"{name}.log"
        entry = {"name": name, "cmd": cmd, "started": datetime.now().isoformat(), "log": str(log)}
        status["jobs"].append(entry)
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"=== [{name}] {cmd}", flush=True)
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, shell=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})  # 로그 즉시 기록 (epoch 진행 확인용)
        entry.update({"returncode": proc.returncode, "seconds": round(time.time() - t0), "finished": datetime.now().isoformat()})
        tail = "".join(open(log, encoding="utf-8", errors="replace").readlines()[-8:])
        print(f"=== [{name}] rc={proc.returncode} {entry['seconds']}s\n{tail}", flush=True)
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    status["finished"] = datetime.now().isoformat()
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [j["name"] for j in status["jobs"] if j.get("returncode")]
    print("QUEUE DONE. failed:", failed or "none")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
