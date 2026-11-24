#!/usr/bin/env python3
"""hy4_rel.py MODEL TOKDIR BIN_0220 BIN_0221 OUTDIR RESULTS_TSV : Hy4-Flash short run of the RELEASED strata-hy4 SYCL packages (0.1.66 / 0.1.67) on the B70.

1. greedy "The capital of France is" (32 tokens, --serve protocol) for --draft off and --draft dfly2 on both releases: the text must contain "Paris",
   the ids are compared between the releases (record only);
2. 1024-token prompt, 64 new tokens, --draft off, alternating 0.2.20 / 0.2.21 for 2 rounds (prefill ppts, decode tok/s).
Appends rows "hy4 <release> <test> <value>" to RESULTS_TSV and prints RESULT lines. Exit 1 when a run failed or "Paris" is missing. Loopback / local only."""
import hashlib
import json
import os
import re
import subprocess
import sys
import time

model, tokdir, bin20, bin21, outdir, tsv = sys.argv[1:7]
os.makedirs(outdir, exist_ok=True)
BINS = [("0.2.20", bin20), ("0.2.21", bin21)]
ARGS = ["--no-engram", "--vram-experts-gb", "18", "--ram-experts-gb", "14"]
bad = 0


def row(rel, test, val):
    val = str(val).replace("\t", " ").replace("\n", " ")
    print(f"RESULT hy4 {rel} | {test} | {val}", flush=True)
    with open(tsv, "a", encoding="utf-8") as f:
        f.write(f"hy4\t{rel}\t{test}\t{val}\n")


tj = json.load(open(os.path.join(tokdir, "tokenizer.hf.json"), encoding="utf-8"))
vocab = dict(tj["model"]["vocab"])
for at in tj.get("added_tokens", []):
    vocab[at["content"]] = at["id"]
id2tok = {v: k for k, v in vocab.items()}


def b2u():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


u2b = {v: k for k, v in b2u().items()}


def decode_ids(ids):
    raw = bytearray()
    for t in ids:
        s = id2tok.get(int(t), "<%s>" % t)
        if s.startswith("<") and s.endswith(">"):
            raw.extend(s.encode("utf-8"))
        else:
            for ch in s:
                raw.extend([u2b[ch]] if ch in u2b else ch.encode("utf-8"))
    return raw.decode("utf-8", errors="replace")


prompt_ids = [vocab[w] for w in ["The", "Ġcapital", "Ġof", "ĠFrance", "Ġis"]]
seed = [vocab[w] for w in ["def", "Ġcompute", "(", "x", ",", "Ġy", "):", "Ċ", "ĠĠĠĠ", "return", "Ġx", "Ġ+", "Ġy", "Ċ"] if w in vocab]
p1024 = (seed * (1024 // len(seed) + 1))[:1024]
open(os.path.join(outdir, "p1024.ids"), "w").write(" ".join(map(str, p1024)) + "\n")


def serve_greedy(binp, rel, draft):
    global bad
    env = dict(os.environ)
    cmd = [binp, "--serve", "--native", model, "--max-context", "4096"] + ARGS + ["--draft", draft]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.path.join(outdir, f"serve-{rel}-{draft}.err"), "w"), text=True, bufsize=1, env=env)
    ready = False
    for line in p.stdout:
        if line.startswith("READY "):
            ready = True
            break
    if not ready:
        rc = p.wait(timeout=60)
        row(rel, f"greedy32 {draft}", f"EXITED BEFORE READY rc={rc}")
        bad += 1
        return
    t0 = time.time()
    p.stdin.write("GEN 32 temperature=0.0 %s\n" % ",".join(map(str, prompt_ids)))
    p.stdin.flush()
    out, done = [], ""
    for ln in p.stdout:
        ln = ln.rstrip("\n")
        if ln.startswith("T "):
            out.append(int(ln.split()[1]))
        elif ln.startswith("DONE "):
            done = ln
            break
        elif ln.startswith("ERR "):
            done = ln
            break
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.stdin.close()
    try:
        p.wait(timeout=120)
    except subprocess.TimeoutExpired:
        p.kill()
    txt = decode_ids(out)
    ok = "Paris" in txt
    print(f"[{rel} {draft}] {done} text={txt!r} ({time.time() - t0:.0f}s)", flush=True)
    row(rel, f"greedy32 {draft} ids md5", hashlib.md5(",".join(map(str, out)).encode()).hexdigest()[:12])
    row(rel, f"greedy32 {draft} Paris", "yes" if ok else "NO")
    if not ok:
        bad += 1


for draft in ("off", "dfly2"):
    for rel, b in BINS:
        serve_greedy(b, rel, draft)

for rnd in (1, 2):
    order = BINS if rnd == 1 else BINS[::-1]
    for rel, b in order:
        cmd = [b, "--model", model, "--ctx", "4096", "--tokens", os.path.join(outdir, "p1024.ids"), "--out", outdir, "--gen", "64", "--draft", "off"] + ARGS
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        log = r.stdout + "\n" + r.stderr
        open(os.path.join(outdir, f"speed-{rel}-r{rnd}.log"), "w").write(log)
        m1 = re.search(r"\[prefill\] total [\d.]+ s = ([\d.]+) ppts", log)
        m2 = re.search(r"\[decode\] \d+ tokens in [\d.]+ s = ([\d.]+) tok/s", log)
        if r.returncode != 0 or not (m1 and m2):
            row(rel, f"1k prefill+decode r{rnd}", f"FAILED rc={r.returncode} " + log[-200:])
            bad += 1
            continue
        row(rel, "1k prompt prefill ppts", m1.group(1))
        row(rel, "1k prompt decode tok/s", m2.group(1))
sys.exit(1 if bad else 0)
