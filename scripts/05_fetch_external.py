#!/usr/bin/env python3
"""05_fetch_external.py — download a SUBSET of the public COW160x4 Cowrie dataset.

Dataset : Mikulas, Krajcik, Helebrandt, Kotuliak (2026). "Deployment of 160 Cowrie SSH
          honeypot instances across four distinct configurations". Zenodo.
          DOI 10.5281/zenodo.21260400  (CC BY 4.0)
Archive : data_all.zip (~19.4 GB) holds one gzipped JSON-lines file per day (*.json.gz).

The lab VM has a 100 GB disk, so the whole archive is NOT downloaded. remotezip reads the
zip's central directory with HTTP Range requests and extracts only the chosen day files.

Usage (Ubuntu defender VM):
    pip install remotezip
    python3 scripts/05_fetch_external.py --list
    python3 scripts/05_fetch_external.py --first 2 --out data/external/cow160
    python3 scripts/05_fetch_external.py --members <name1> <name2> --out data/external/cow160
"""
import argparse
import hashlib
import json
import os
import sys
import time

DEFAULT_URL = "https://zenodo.org/records/21260400/files/data_all.zip?download=1"


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--list", action="store_true", help="list day files and sizes, then exit")
    ap.add_argument("--first", type=int, default=0, help="extract the first N day files (sorted)")
    ap.add_argument("--members", nargs="*", default=[], help="explicit member names to extract")
    ap.add_argument("--out", default="data/external/cow160")
    ap.add_argument("--max-gb", type=float, default=3.0, help="refuse if selection exceeds this")
    a = ap.parse_args()

    try:
        from remotezip import RemoteZip, RemoteIOError
    except ImportError:
        sys.exit("[-] remotezip missing:  pip install remotezip")

    try:
        z = RemoteZip(a.url)
    except RemoteIOError:
        # some servers reject suffix ranges ("bytes=-N"); fall back to explicit ranges
        z = RemoteZip(a.url, support_suffix_range=False)
    with z:
        infos = sorted([i for i in z.infolist() if i.filename.endswith(".json.gz")],
                       key=lambda i: i.filename)
        if a.list or (not a.first and not a.members):
            for i in infos:
                print(f"{i.filename:45s} {i.compress_size / 2**20:9.1f} MB")
            print(f"[i] {len(infos)} day files")
            return
        chosen = infos[:a.first] if a.first else [i for i in infos if i.filename in set(a.members)]
        if not chosen:
            sys.exit("[-] nothing selected (check names with --list)")
        total_gb = sum(i.compress_size for i in chosen) / 2**30
        if total_gb > a.max_gb:
            sys.exit(f"[-] selection is {total_gb:.2f} GB > --max-gb {a.max_gb}; pick fewer days")
        os.makedirs(a.out, exist_ok=True)
        manifest = {"source_url": a.url, "doi": "10.5281/zenodo.21260400",
                    "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "files": []}
        for i in chosen:
            print(f"[>] {i.filename} ({i.compress_size / 2**20:.1f} MB)")
            path = z.extract(i.filename, a.out)
            manifest["files"].append({"member": i.filename, "path": path,
                                      "bytes": os.path.getsize(path),
                                      "crc32_zip": f"{i.CRC:08x}",
                                      "sha256": _sha256(path)})
        json.dump(manifest, open(os.path.join(a.out, "MANIFEST.json"), "w"), indent=2)
        print(f"[+] {len(chosen)} files -> {a.out}  (manifest: MANIFEST.json)")


if __name__ == "__main__":
    main()
