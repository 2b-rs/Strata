#!/usr/bin/env python3
"""verify_component.py DIR : check every file listed in DIR/engine-component.json (sha256 and byte count). Exit 0 only when all match."""
import hashlib
import json
import os
import sys

d = sys.argv[1]
comp = json.load(open(os.path.join(d, "engine-component.json")))
bad = 0
for f in comp.get("files", []):
    p = os.path.join(d, f["path"])
    if not os.path.isfile(p):
        print("MISSING", f["path"])
        bad += 1
        continue
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != f.get("sha256") or (f.get("bytes") is not None and os.path.getsize(p) != f["bytes"]):
        print("MISMATCH", f["path"])
        bad += 1
print("component %s %s: %d files, %d bad" % (comp.get("id"), comp.get("version"), len(comp.get("files", [])), bad))
sys.exit(1 if bad else 0)
