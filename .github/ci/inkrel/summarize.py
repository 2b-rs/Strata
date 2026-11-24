#!/usr/bin/env python3
"""summarize.py RESULTS_TSV [OUT_MD] : pivot results.tsv (model <TAB> release <TAB> test <TAB> value, one row per measurement) into a markdown table,
one row per (model, test) with the 0.2.20 and 0.2.21 values (all rounds, then the median for numbers) and the ratio of the medians.
Prints the table; also writes it to OUT_MD and to $GITHUB_STEP_SUMMARY when set."""
import collections
import os
import statistics
import sys

rows = collections.OrderedDict()
for ln in open(sys.argv[1], encoding="utf-8", errors="replace"):
    p = ln.rstrip("\n").split("\t")
    if len(p) < 4:
        continue
    model, rel, test, val = p[0], p[1], p[2], "\t".join(p[3:])
    rows.setdefault((model, test), collections.OrderedDict()).setdefault(rel, []).append(val)


def num(v):
    try:
        return float(v)
    except ValueError:
        return None


def cell(vals):
    if not vals:
        return "-"
    ns = [num(v) for v in vals]
    if all(n is not None for n in ns):
        med = statistics.median(ns)
        return "%s (median %.4g)" % (" / ".join(vals), med) if len(vals) > 1 else vals[0]
    return " / ".join(dict.fromkeys(vals))


out = ["| model | test | 0.2.20 (0.1.66) | 0.2.21 (0.1.67) | 0.2.21 / 0.2.20 |", "|---|---|---|---|---|"]
for (model, test), by in rows.items():
    a, b = by.get("0.2.20", []), by.get("0.2.21", [])
    ratio = ""
    na, nb = [num(v) for v in a], [num(v) for v in b]
    if a and b and all(x is not None for x in na + nb):
        ma, mb = statistics.median(na), statistics.median(nb)
        if ma:
            ratio = "%.3f" % (mb / ma)
    out.append("| %s | %s | %s | %s | %s |" % (model, test, cell(a).replace("|", "/"), cell(b).replace("|", "/"), ratio))
txt = "\n".join(out) + "\n"
print(txt)
if len(sys.argv) > 2:
    open(sys.argv[2], "w", encoding="utf-8").write(txt)
gs = os.environ.get("GITHUB_STEP_SUMMARY")
if gs:
    with open(gs, "a", encoding="utf-8") as f:
        f.write("## B70 release test: 0.2.20 against 0.2.21\n\n" + txt)
