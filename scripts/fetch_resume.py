#!/usr/bin/env python3
"""fetch_resume.py - download chosen day files from the COW160x4 archive (Zenodo
10.5281/zenodo.21260400, data_all.zip) with resume and retries, then check each file's CRC-32
against the archive's central directory. Safe on unstable connections: a broken transfer
continues from the last byte instead of starting again.

Usage:
  python3 scripts/fetch_resume.py --out data/external/cow160/confirm/logs_by_day \
      2025-07-03 2025-07-14 2025-07-26 2025-07-31 2025-08-10
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import time
import zlib

import requests
from remotezip import RemoteZip

URL = "https://zenodo.org/records/21260400/files/data_all.zip?download=1"
CHUNK = 1 << 20


def get_range(url, start, end, out, tries=200):
    """Append bytes [start, end] of url to file out, resuming after any error."""
    for attempt in range(1, tries + 1):
        have = os.path.getsize(out) if os.path.exists(out) else 0
        if start + have > end:
            return
        try:
            r = requests.get(url, headers={"Range": f"bytes={start + have}-{end}"},
                             stream=True, timeout=60)
            if r.status_code != 206:
                raise IOError(f"HTTP {r.status_code} (range not honoured)")
            with open(out, "ab") as f:
                for b in r.iter_content(CHUNK):
                    f.write(b)
                    have += len(b)
                    print(f"\r    {have / 2**20:8.1f} / {(end - start + 1) / 2**20:.1f} MB",
                          end="", flush=True)
        except (requests.RequestException, IOError) as e:
            print(f"\n    [!] attempt {attempt}: {e} - retrying in 10 s", flush=True)
            time.sleep(10)
    if os.path.getsize(out) < end - start + 1:
        sys.exit("[-] gave up after many retries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("days", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=URL)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with RemoteZip(a.url, support_suffix_range=False) as z:
        info = {os.path.basename(i.filename)[:10]: i for i in z.infolist()
                if i.filename.endswith(".json.gz")}
    manifest = []
    for day in a.days:
        i = info.get(day)
        if i is None:
            sys.exit(f"[-] {day} not in the archive")
        final = os.path.join(a.out, f"{day}.json.gz")
        part = final + ".part"
        print(f"[>] {day}  ({i.compress_size / 2**20:.1f} MB, method {i.compress_type})")
        if os.path.exists(final) and zlib.crc32(open(final, "rb").read()) & 0xFFFFFFFF == i.CRC:
            print("    already complete (CRC ok)")
        else:
            # local header: 30 bytes + name + extra; its lengths are read from the header itself
            hdr = final + ".hdr"
            if os.path.exists(hdr):
                os.remove(hdr)
            get_range(a.url, i.header_offset, i.header_offset + 29, hdr)
            raw = open(hdr, "rb").read()
            os.remove(hdr)
            if raw[:4] != b"PK\x03\x04":
                sys.exit("[-] bad local header")
            n, m = struct.unpack("<HH", raw[26:30])
            start = i.header_offset + 30 + n + m
            get_range(a.url, start, start + i.compress_size - 1, part)
            print()
            data = open(part, "rb").read()
            if len(data) != i.compress_size:
                sys.exit(f"[-] {day}: size {len(data)} != {i.compress_size}")
            if i.compress_type == 8:
                data = zlib.decompressobj(-15).decompress(data)
            elif i.compress_type != 0:
                sys.exit(f"[-] unsupported compression {i.compress_type}")
            if zlib.crc32(data) & 0xFFFFFFFF != i.CRC:
                os.remove(part)
                sys.exit(f"[-] {day}: CRC mismatch - partial file removed, run again")
            with open(final, "wb") as f:
                f.write(data)
            os.remove(part)
            print("    CRC ok")
        manifest.append({"day": day, "member": i.filename, "crc32": f"{i.CRC:08x}",
                         "sha256": hashlib.sha256(open(final, "rb").read()).hexdigest()})
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump({"source_url": a.url, "doi": "10.5281/zenodo.21260400",
                   "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "files": manifest}, f, indent=1)
    for x in manifest:
        print(f"{x['sha256']}  {x['day']}.json.gz")
    print(f"[+] {len(manifest)} file(s) complete; manifest.json written")


if __name__ == "__main__":
    main()
