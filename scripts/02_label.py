#!/usr/bin/env python3
"""02_label.py — assign ground-truth label from PROVENANCE (login username), not features."""
import argparse
import pandas as pd


def label(df):
    u = df.login_username.astype("string").fillna("")
    is_p = u.str.match(r"^p\d+$")
    is_bot = u.str.match(r"^bot\d+$")
    df = df.copy()
    df["is_human"] = is_p.astype(int)
    df["labeled"] = (is_p | is_bot)
    df["group"] = u.where(is_p | is_bot, other="attack_tool")
    df["population"] = df.command_count.apply(
        lambda c: "interactive" if c >= 1 else "non_interactive")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/sessions.csv")
    ap.add_argument("--out", default="data/processed/labeled.csv")
    a = ap.parse_args()
    df = label(pd.read_csv(a.inp))
    df.to_csv(a.out, index=False)
    lab = df[df.labeled]
    print(f"[+] labeled {len(lab)} controlled sessions "
          f"(Human={int(lab.is_human.sum())}, Bot={int((lab.is_human==0).sum())})")
    print(f"    interactive controlled sessions: "
          f"{len(lab[lab.population=='interactive'])}")
    print(f"    non-interactive attack-tool sessions (separate): "
          f"{len(df[(df.group=='attack_tool')])}")


if __name__ == "__main__":
    main()
