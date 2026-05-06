# Design and Implementation of an Autonomous Context-Aware Deception Framework

**Semester Project — Cybersecurity**  
Department of Intelligent Information Systems Security Engineering  
Faculty of Artificial Intelligence Engineering — Syrian Private University

| | |
|---|---|
| **Students** | Omar Khaled Abdulkafi / Moaz Hassan Al-Dawalibi |
| **Supervisors** | Dr. Wassiem Jnidi / Eng. Amjad Al-Safadi |
| **Phase** | Semester Project (Phase 1 of 3) |
| **Status** |  Complete |
| **Academic Year** | 2025–2026 |

---

## Overview

An intelligent cybersecurity system that **automatically**:

1.  **Monitors** all network traffic in real-time — Zeek 8.0.5 NSM
2.  **Deceives** attackers via a realistic fake SSH server — Cowrie honeypot
3.  **Analyzes** its own exposed attack surface — Nmap + Python ASA module
4.  **Classifies** each attacker as **Bot** or **Human** — Random Forest ML
5.  **Validates** against real-world data — UWF-ZeekData22 dataset

> **Why "Context-Aware"?**  
> The system scans its own exposed services, discovers that SSH (port 22) is exposed, and autonomously deploys a Cowrie SSH honeypot — a decision driven by observed context, not static configuration.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DEFENDER NODE (Ubuntu 22.04)             │
│  IP: 192.168.211.128                                        │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Monitoring   │  │  Deception   │  │  Analysis        │  │
│  │ Layer        │  │  Layer       │  │  Layer           │  │
│  │              │  │              │  │                  │  │
│  │  Zeek 8.0.5  │  │  Cowrie SSH  │  │  Random Forest   │  │
│  │  conn.log    │  │  Honeypot    │  │  Classifier      │  │
│  │  ssh.log     │  │  port 2222   │  │  Bot vs Human    │  │
│  └──────────────┘  └──────┬───────┘  └──────────────────┘  │
│                           │                                  │
│         iptables: port 22 → 2222                            │
└───────────────────────────┼─────────────────────────────────┘
                            │  VMnet8 NAT 192.168.211.0/24
┌───────────────────────────┼─────────────────────────────────┐
│                    ATTACKER NODE (Kali 2025)                 │
│  IP: 192.168.211.129                                        │
│  Nmap · Hydra · Python Bot · Manual SSH · Metasploit        │
└─────────────────────────────────────────────────────────────┘
```

---

## Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 97.9% |
| **AUC** | 0.9942 |
| **False Negatives (missed attacks)** | **0** |
| **CV F1 (5-fold)** | 0.978 ± 0.025 |
| **External validation FN (UWF-ZeekData22)** | **0** |

---

## Project Structure

```
deception-framework/
├── scripts/
│   ├── collect_cowrie.py       ← Phase 1: collect Cowrie logs
│   ├── collect_zeek.py         ← Phase 1: collect Zeek conn.log
│   ├── build_dataset.py        ← Phase 2: engineer features + label
│   ├── train_model.py          ← Phase 3: train + evaluate RF
│   ├── asa_parser.py           ← Phase 4: parse Nmap XML → risk report
│   └── validate_uwf.py         ← Phase 5: external dataset validation
├── data/
│   ├── raw/                    ← conn_raw.csv, cowrie_raw.csv
│   └── processed/              ← labeled_dataset.csv, uwf_dataset.csv
├── models/                     ← rf_model.pkl, scaler.pkl
├── reports/                    ← confusion_matrix.png, roc_curve.png,
│                                   feature_importance.png, icd_comparison.png
├── docs/
│   └── attack_surface_report.csv
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/deception-framework.git
cd deception-framework

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Ensure Zeek and Cowrie are running on the Defender Node
sudo /opt/zeek/bin/zeekctl status
sudo systemctl status cowrie
```

---

## Usage

### Step 1 — Collect raw data
```bash
python3 scripts/collect_cowrie.py    # reads all cowrie.json files
python3 scripts/collect_zeek.py      # reads Zeek conn.log
```

### Step 2 — Build labeled dataset
```bash
python3 scripts/build_dataset.py
# Output: data/processed/labeled_dataset.csv
# Prints: session count, Bot count, Human count
```

### Step 3 — Train and evaluate the model
```bash
python3 scripts/train_model.py
# Output: models/rf_model.pkl, models/scaler.pkl
#         reports/confusion_matrix.png
#         reports/roc_curve.png
#         reports/feature_importance.png
#         reports/icd_comparison.png
```

### Step 4 — Run attack surface analysis
```bash
# On Kali — save Nmap output first:
# sudo nmap -sV -sC -O -p- 192.168.211.128 -oX scan.xml

python3 scripts/asa_parser.py --input scan.xml --output docs/attack_surface_report.csv
```

---

## Features (18+)

| # | Feature | Source | Bot Range | Human Range |
|---|---------|--------|-----------|-------------|
| 1 | icd_mean ⭐ | cowrie.json | 0.01–0.5 s | 3–15 s |
| 2 | unique_commands ⭐ | cowrie.json | 0–2 | 5–15 |
| 3 | command_count | cowrie.json | 0–3 | 5–20 |
| 4 | command_diversity | cowrie.json | 0.0–0.3 | 0.6–1.0 |
| 5 | failed_logins | cowrie.json | 5–500 | 0–2 |
| 6 | success_logins | cowrie.json | 0–1 | 1 |
| 7 | session_duration | cowrie.json | 1–10 s | 20–600 s |
| 8 | icd_std | cowrie.json | 0.001–0.1 | 1–8 |

> **icd_mean** = Inter-Command Delay Mean — bots act in milliseconds; humans pause to think.

---

## Attack Scenarios

| Scenario | Tool | Class | Key Signature |
|----------|------|-------|---------------|
| Network Reconnaissance | Nmap | Bot | High connection rate, no login |
| SSH Brute Force | Hydra | Bot | Hundreds of failed logins |
| Automated Bot Session | Python + Paramiko | Bot | icd_mean < 0.5s |
| Manual Human Session | Interactive SSH | Human | icd_mean > 4s, diverse commands |
| Framework Scan | Metasploit | Bot | Structured probe, distinctive client |

---

## References

1. Bagui et al., "UWF-ZeekData22," *Data*, MDPI, 2023
2. Kocak & Yilmaz, "RF-Based NIDS," *JNSM*, Springer, 2024
3. Moric et al., "Honeypots and Deception," *Informatics*, MDPI, 2025
4. Singh, "SSH Attack Patterns using Cowrie," *IJCA*, 2026
5. Franco et al., "Survey of Honeypots," *IEEE COMST*, 2021

---

## Roadmap

```
Semester Project (✅ Complete)
  └── Zeek + Cowrie + ASA + Random Forest + UWF Validation

GP1 — Graduation Project 1 (Planned)
  └── Docker dynamic decoys + Python Orchestrator + TinyLlama LLM + MTD

GP2 — Graduation Project 2 (Planned)
  └── Markov Chain intent prediction + SHAP/XAI + Streamlit dashboard
```

---

*Syrian Private University — Faculty of Artificial Intelligence Engineering — 2025/2026*
