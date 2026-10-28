#!/usr/bin/env python3
"""test_ingest.py - the live readers on temporary files (the real logs are never touched).
Run:  python3 orchestrator/tests/test_ingest.py   (or with pytest)"""
import json
import os
import queue
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest_cowrie import CowrieReader  # noqa: E402
from ingest_zeek import ZeekConnReader  # noqa: E402
from tail import Tail  # noqa: E402

ZEEK_HDR = ("#separator \\x09\n#set_separator\t,\n#path\tconn\n"
            "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tservice\t"
            "duration\tconn_state\thistory\n")


def tmpfile(text=""):
    d = tempfile.mkdtemp(prefix="acdf_ingest_")
    p = os.path.join(d, "log")
    with open(p, "w") as f:
        f.write(text)
    return p


def collect(gen_owner, n, timeout=5.0):
    """Run gen_owner(stop) in a thread, return the first n items it puts on its queue."""
    q, stop = gen_owner.q, threading.Event()
    threading.Thread(target=gen_owner.run, args=(stop,), daemon=True).start()
    out, end = [], time.time() + timeout
    while len(out) < n and time.time() < end:
        try:
            out.append(q.get(timeout=0.1))
        except queue.Empty:
            pass
    stop.set()
    return out


def append(p, text):
    with open(p, "a") as f:
        f.write(text)
        f.flush()


def cev(eid, session="s1", t="2026-10-18T21:35:25.000000Z", **kw):
    return json.dumps({"eventid": eid, "session": session, "timestamp": t, **kw}) + "\n"


def test_tail_waits_for_complete_lines():
    p = tmpfile()
    t = Tail(p, from_start=True, poll=0.02)
    got, stop = [], threading.Event()

    def run():
        for line in t.lines(stop):
            got.append(line)
    threading.Thread(target=run, daemon=True).start()
    append(p, "abc")                 # half a line ...
    time.sleep(0.2)
    assert got == []
    append(p, "def\nxyz\n")          # ... completed later
    time.sleep(0.3)
    stop.set()
    assert got == ["abcdef", "xyz"]


def test_cowrie_skips_existing_and_malformed_lines():
    p = tmpfile(cev("cowrie.command.input", input="old"))   # already in the file: not emitted
    r = CowrieReader(queue.Queue(), p)
    r.tail.poll = 0.02
    threading.Timer(0.2, append, (p, "{broken json\n" + cev("cowrie.session.connect", src_ip="10.0.0.5")
                                  + cev("cowrie.direct-tcpip.data") + cev("cowrie.command.input", input="ls")
                                  )).start()
    got = collect(r, 2)
    assert [e[2]["eventid"] for e in got] == ["cowrie.session.connect", "cowrie.command.input"]
    assert got[1][2]["input"] == "ls" and abs(got[0][1] - 1792359325.0) < 1e-6
    assert r.n_bad == 1 and r.n_skipped == 1


def test_cowrie_survives_rotation():
    p = tmpfile()
    r = CowrieReader(queue.Queue(), p)
    r.tail.poll = 0.02

    def rotate():
        append(p, cev("cowrie.command.input", input="before"))
        time.sleep(0.2)
        os.rename(p, p + ".2026-10-18")           # what Cowrie does at midnight
        with open(p, "w") as f:
            f.write(cev("cowrie.command.input", input="after"))
    threading.Timer(0.1, rotate).start()
    got = collect(r, 2)
    assert [e[2]["input"] for e in got] == ["before", "after"] and r.tail.rotations == 1


def test_zeek_tsv_and_rotation():
    p = tmpfile(ZEEK_HDR + "1760831000.1\tCold\t1.1.1.1\t1\t2.2.2.2\t2\ttcp\t-\t-\tSF\tSh\n")
    r = ZeekConnReader(queue.Queue(), p)
    r.tail.poll = 0.02
    assert r.fields[2] == "id.orig_h" and r.sep == "\t"

    def later():
        append(p, "1760831005.5\tC1\t192.168.71.129\t51000\t192.168.71.128\t3306\ttcp\t-\t-\tREJ\tSr\n")
        time.sleep(0.2)
        os.rename(p, p + ".old")                  # hourly rotation by zeekctl
        with open(p, "w") as f:
            f.write(ZEEK_HDR + "1760834600.0\tC2\t192.168.71.129\t51001\t192.168.71.128\t8080"
                               "\ttcp\t-\t-\tS0\tS\n")
    threading.Timer(0.1, later).start()
    got = collect(r, 2)
    c1, c2 = got[0][2], got[1][2]
    assert (c1["id.resp_p"], c1["conn_state"], c1["service"]) == (3306, "REJ", None)
    assert (c2["id.resp_p"], c2["conn_state"]) == (8080, "S0") and r.tail.rotations == 1
    assert r.n_events == 2                        # the old record (Cold) was not emitted


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"[+] {len(tests)} tests passed")
