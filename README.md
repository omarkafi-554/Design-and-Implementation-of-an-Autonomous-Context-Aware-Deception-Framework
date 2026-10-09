# Autonomous Context-Aware Deception Framework

**Semester Project — corrected scientific pipeline**
Classifying SSH honeypot sessions as **Bot** or **Human** from behavioural and keystroke-dynamics features, in an isolated two-VM lab.

Authors: Omar Khaled Abdulkafi, Moaz Hassan Al-Dawalibi
Supervisors: Dr. Wassiem Jnidi, Eng. Amjad Al-Safadi

---

## What this project does

An SSH honeypot (Cowrie) behind a transparent `iptables` redirect captures attacker sessions; Zeek monitors the network; Nmap performs attack-surface analysis. A supervised classifier then labels each **interactive** session (one that reached a shell and ran at least one command) as automated (Bot) or human (Human). This repository contains the **corrected** pipeline.

## Ground-truth labelling (no circularity)

Labels come from each session's **provenance — the login username**, recorded by Cowrie at authentication time, independently of any feature:

| Login username | Class | Source |
|---|---|---|
| p1 … p10 | Human | real participants running manual sessions |
| bot1 … bot4 | Bot | auto_session.py generators (fast/medium/slow/burst) |
| root, admin, … | Bot (non-interactive) | Hydra / Nmap / Metasploit |

## Features (01_features.py)

- Command/timing: command_count, unique_commands, command_diversity, icd_mean, icd_std, session_duration
- Keystroke dynamics (TTY): has_tty, single_char_ratio, keystroke_gap_median, backspaces

## Pipeline

    python3 scripts/01_features.py --cowrie ~/cowrie/var/log/cowrie --tty ~/cowrie/var/lib/cowrie/tty --out data/processed/sessions.csv
    python3 scripts/02_label.py   --in data/processed/sessions.csv --out data/processed/labeled.csv
    python3 scripts/03_evaluate.py --in data/processed/labeled.csv --out reports/metrics.json
    python3 scripts/04_train_final.py --in data/processed/labeled.csv --out reports --models models

Bot generator (run on Kali): python3 auto_session.py --user bot1 --profile fast

## Evaluation

Leave-one-group-out (each fold holds out a whole participant or bot profile); primary metric MCC. Honest finding: timing alone is not sufficient — a deliberately slow bot (bot3) falls inside the human timing range and defeats timing-only tree models; keystroke-dynamics features separate the classes.

## Methodology notes (what was corrected)

- Circular labelling removed: labels were computed from icd_mean (a training feature); now from provenance.
- Invalid external validation removed: UWF-ZeekData22 is network-flow data, not honeypot sessions; dropped. Same-type external validation planned for GP1.
- Bots corrected: old bot used exec_command (one command, no timing); auto_session.py opens a real interactive shell.
- Honest limitations: small interactive sample; simple bot generators; keystroke features need TTY logs; adversarial PTY-using human-paced bot is GP1 red-team work.

## Reproducibility

    numpy==2.2.6
    pandas==2.3.3
    scikit-learn==1.7.2
    paramiko>=3.0.0

## Ethics

All offensive activity was performed inside an isolated lab against hosts owned by the project team, with no route to the public internet.
