#!/usr/bin/env python3
"""
Generate a balanced chess dataset file from a labeled source file.
Default: read `data/chess-data-1.txt` and write `data/chess-data-balance.txt` (~15000 lines,
40% Nothing, 40% Check, 20% Checkmate).

Usage:
    python3 tools/generate_chess_data_balance.py \
        --input data/chess-data-1.txt \
        --output data/chess-data-balance.txt \
        --total 15000

"""
import argparse
from collections import Counter
from pathlib import Path
import random
import sys

LABELS = ("Nothing", "Check", "Checkmate")


def load_by_label(src_path: Path):
    text = src_path.read_text(encoding="utf-8")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    groups = {lab: [] for lab in LABELS}
    for ln in lines:
        if " " not in ln:
            continue
        label = ln.rsplit(" ", 1)[-1]
        if label in groups:
            groups[label].append(ln)
    return groups


def main():
    p = argparse.ArgumentParser(description="Generate balanced chess data file")
    p.add_argument(
        "--input", "-i", default="data/chess-data-1.txt", help="source labeled file"
    )
    p.add_argument(
        "--output", "-o", default="data/chess-data-balance.txt", help="output file"
    )
    p.add_argument(
        "--total", "-t", type=int, default=15000, help="approx total lines to generate"
    )
    p.add_argument("--seed", type=int, default=None, help="random seed (optional)")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    src = Path(args.input)
    out = Path(args.output)
    if not src.exists():
        print(f"ERROR: source file not found: {src}", file=sys.stderr)
        sys.exit(2)

    groups = load_by_label(src)
    missing = [lab for lab, lst in groups.items() if not lst]
    if missing:
        print("ERROR: missing examples for labels:", ",".join(missing), file=sys.stderr)
        sys.exit(3)

    total = args.total
    # compute counts: 40% Nothing, 40% Check, 20% Checkmate
    cnt_nothing = int(total * 0.4)
    cnt_check = int(total * 0.4)
    cnt_checkmate = total - (cnt_nothing + cnt_check)
    counts = {"Nothing": cnt_nothing, "Check": cnt_check, "Checkmate": cnt_checkmate}

    # sample with replacement from each label bucket
    out_lines = []
    for lab, k in counts.items():
        out_lines.extend(random.choices(groups[lab], k=k))

    random.shuffle(out_lines)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    ctr = Counter()
    for ln in out_lines:
        ctr[ln.rsplit(" ", 1)[1]] += 1

    print(f"WROTE {out}")
    for lab in LABELS:
        print(lab, ctr[lab])
    print("TOTAL", sum(ctr.values()))


if __name__ == "__main__":
    main()
