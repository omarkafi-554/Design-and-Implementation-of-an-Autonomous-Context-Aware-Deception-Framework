#!/usr/bin/env python3
"""bot_v2.py - lab bot generator v2 (runs on the Kali VM). Replaces auto_session.py.

One username per behaviour class (the provenance label and the LOGO group):
  bot_exec   one exec request: 1-6 commands joined with ';' (never reaches line k)
  bot_rtt    interactive shell; waits for the prompt, then a per-session network delay
             (0.05-0.8 s, +-10 % jitter) before the next line
  bot_sleep  interactive; fixed wait D in {2, 5, 10, 20} s after every line (+-1 %),
             like hard-coded timeouts
  bot_slow   interactive; waits for the prompt, then service time x slowness, where the
             service time depends on the command (0.3 / 1 / 3 s) and slowness is drawn
             per session from 1-10 (log-uniform), +-20 % jitter -> irregular 0.3-30 s gaps
  bot_burst  interactive; writes 2-5 lines at once without waiting (pipelined)
  bot_mimic  RED-TEAM ONLY (never used for training): types character by character with
             log-normal key gaps (median 0.18 s), think time before each line (median 4 s)
             and 5 % typos corrected with backspace
Every session draws 1-15 commands (bot_burst 3-15) without replacement from a generic pool
of 35 commands (4 of them contain a $(...) substitution, so the Cowrie artefact is exercised)
and ends with 'exit', sent with the same class timing as the other lines.

Usage (Kali):
  python3 bot_v2.py --host 192.168.71.128 --port 22 --cls bot_rtt --sessions 20 --seed 1
"""
import argparse
import random
import re
import socket
import time

import paramiko

POOL = [  # (command, cost: 0 cheap / 1 medium / 2 heavy)
    ("uname -a", 0), ("id", 0), ("whoami", 0), ("hostname", 0), ("uptime", 0), ("w", 0),
    ("cat /etc/os-release", 0), ("cat /proc/cpuinfo | grep name | head -n 1", 1),
    ("free -m", 0), ("df -h", 1), ("ls -la /tmp", 0), ("ps aux | head -n 15", 1),
    ("last | head -n 5", 1), ("tail -n 5 /etc/passwd", 0), ("crontab -l", 0),
    ("ifconfig", 1), ("ip a", 1), ("netstat -tlnp", 2), ("nproc", 0),
    ("lscpu | head -n 5", 1), ("ls -la ~/.ssh", 0), ("cat ~/.bash_history | tail -n 5", 0),
    ("env | head -n 5", 0), ("mount | head -n 3", 1), ("which wget curl python3 perl", 0),
    ("cd /tmp && ls -la", 0), ("cat /etc/hostname", 0), ("ls /var/log", 0),
    ("find / -name '*.conf' -maxdepth 2 2>/dev/null | head -n 5", 2),
    ("cat /proc/meminfo | head -n 3", 0), ("echo $(hostname)", 0),
    ("echo $(uname -m) $(nproc)", 0), ("ls -lh $(which ls)", 0),
    ("arch=$(uname -m); echo $arch", 0), ("du -sh /var 2>/dev/null", 2),
]
SERVICE = {0: 0.3, 1: 1.0, 2: 3.0}
PROMPT = re.compile(rb"[#$] $")


def read_until_prompt(chan, timeout=15.0):
    """Read output until the shell prompt ('# ' or '$ ' at the end) or the timeout."""
    buf, end = b"", time.time() + timeout
    while time.time() < end:
        if chan.recv_ready():
            buf += chan.recv(65536)
            if PROMPT.search(buf):
                return buf
        else:
            time.sleep(0.01)
    return buf


def pick(rng, lo, hi):
    return rng.sample(POOL, rng.randint(lo, min(hi, len(POOL))))


def connect(a, user):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(a.host, port=a.port, username=user, password=a.password, timeout=20,
              allow_agent=False, look_for_keys=False, banner_timeout=30)
    return c


def run_exec(c, rng):
    cmds = pick(rng, 1, 6)
    try:
        _, out, _ = c.exec_command("; ".join(x for x, _ in cmds), timeout=30)
        out.read()
    except paramiko.SSHException as e:
        # Cowrie runs the command and closes the channel before paramiko reads the
        # exec reply; the session is complete and logged, so this is not an error.
        if "Channel closed" not in str(e):
            raise


def type_line(chan, rng, cmd):
    """bot_mimic: one keystroke per write, log-normal key gaps, 5 % typos fixed with DEL."""
    for ch in cmd:
        if rng.random() < 0.05:
            chan.send(rng.choice("asdfjkl"))
            time.sleep(rng.lognormvariate(-1.715, 0.4))
            chan.send("\x7f")
            time.sleep(rng.lognormvariate(-1.715, 0.4))
        chan.send(ch)
        time.sleep(rng.lognormvariate(-1.715, 0.4))             # key gap, median 0.18 s
    chan.send("\r")


def run_shell(c, cls, rng):
    """'exit' is the last line and follows the same class timing as every other line."""
    chan = c.invoke_shell(term="xterm", width=120, height=40)
    read_until_prompt(chan)
    cmds = pick(rng, 3 if cls == "bot_burst" else 1, 15) + [("exit", 0)]
    rtt = rng.uniform(0.05, 0.8)
    fixed = rng.choice([2, 5, 10, 20])
    slowness = 10 ** rng.uniform(0, 1)
    if cls == "bot_burst":
        i = 0
        while i < len(cmds):
            n = rng.randint(2, 5)
            chan.send("".join(x + "\n" for x, _ in cmds[i:i + n]))
            i += n
            if i < len(cmds):
                read_until_prompt(chan, timeout=5)
    else:
        prev_cost = 0
        for j, (cmd, cost) in enumerate(cmds):
            if cls == "bot_mimic":
                time.sleep(rng.lognormvariate(1.386, 0.6))      # think time, median 4 s
            elif j and cls == "bot_rtt":
                time.sleep(rtt * rng.uniform(0.9, 1.1))
            elif j and cls == "bot_sleep":
                time.sleep(fixed * rng.uniform(0.99, 1.01))
            elif j and cls == "bot_slow":
                time.sleep(SERVICE[prev_cost] * slowness * rng.uniform(0.8, 1.2))
            if cls == "bot_mimic":
                type_line(chan, rng, cmd)
            else:
                chan.send(cmd + "\n")
            if cmd == "exit":
                break
            read_until_prompt(chan)
            prev_cost = cost
    time.sleep(0.5)
    chan.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--cls", required=True, choices=["bot_exec", "bot_rtt", "bot_sleep",
                                                     "bot_slow", "bot_burst", "bot_mimic"])
    ap.add_argument("--sessions", type=int, default=20)
    ap.add_argument("--password", default="botpass")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(f"{a.cls}-{a.seed}")
    for i in range(a.sessions):
        t0 = time.time()
        try:
            c = connect(a, a.cls)
            run_exec(c, rng) if a.cls == "bot_exec" else run_shell(c, a.cls, rng)
            c.close()
            status = "ok"
        except (paramiko.SSHException, socket.error) as e:
            status = f"error: {e}"
        print(f"[{a.cls}] session {i + 1}/{a.sessions} {status} ({time.time() - t0:.1f} s)",
              flush=True)
        time.sleep(rng.uniform(2, 5))


if __name__ == "__main__":
    main()
