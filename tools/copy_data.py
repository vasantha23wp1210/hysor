"""Copy the large data files listed in DATA_MANIFEST.csv (kind == data) from a uatie-edm-phase1 checkout.

    python tools/copy_data.py --from <path-to-uatie-edm-phase1>

Run `git lfs pull` in that checkout first. Every copy is checked against its SHA-256.
"""
import argparse
import csv
import hashlib
import shutil
import sys
from pathlib import Path

HYSOR = Path(__file__).resolve().parents[1]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True, help="path to a uatie-edm-phase1 checkout")
    src_repo = Path(ap.parse_args().src).resolve()
    rows = [r for r in csv.DictReader(open(HYSOR / "DATA_MANIFEST.csv", newline="")) if r["kind"] == "data"]
    failed = []
    for r in rows:
        src, dst = src_repo / r["path"], HYSOR / r["path"]
        if dst.exists() and sha256(dst) == r["sha256"]:
            print(f"ok      {r['path']}")
            continue
        if not src.exists() or sha256(src) != r["sha256"]:
            failed.append(r["path"])
            print(f"FAILED  {r['path']}  (missing or hash differs in source checkout; run `git lfs pull` there)")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"copied  {r['path']}")
    print(f"{len(rows) - len(failed)}/{len(rows)} data files in place")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
