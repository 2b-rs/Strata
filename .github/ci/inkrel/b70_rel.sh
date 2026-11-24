#!/bin/bash
# Strata 0.2.21 against 0.2.20 on the Arc Pro B70 runner, using the PUBLISHED Linux SYCL packages (no build here).
#   b70_rel.sh pkgs                          download + sha256-check + unpack the inkling / hy4 / engine packages of v0.2.20 and v0.2.21, --version of each
#   b70_rel.sh inkling <label> <model-url> <model-file> [<dspark-url> <dspark-file>] [<sha256>]
#   b70_rel.sh hy4                           Hy4-Flash-SQ-64: greedy + 1k speed on both releases (needs 62 GB of disk)
#   b70_rel.sh summary                       pivot table of 0.2.20 against 0.2.21 (printed, step summary)
# Work dir: $RUNNER_TEMP/b70-rel0221 (the workflow removes it at the end). Results: $W/logs/results.tsv (model, release, test, value).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
W=${RUNNER_TEMP:-/tmp}/b70-rel0221
LOGS=$W/logs; PKG=$W/pkg; MODELS=$W/models
TSV=$LOGS/results.tsv
mkdir -p "$LOGS" "$PKG" "$MODELS"
REL_URL=https://github.com/Niko1221/Strata/releases/download
HF_STRATA=https://huggingface.co/strata-engine/Inkling-2-Mini-Strata/resolve/main
HF_UNSLOTH=https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main
HF_HY4=https://huggingface.co/strata-engine/Hy4-Flash-Strata/resolve/main
# release label -> tag, package version
tag_of() { case $1 in 0.2.20) echo v0.2.20;; 0.2.21) echo v0.2.21;; esac; }
ver_of() { case $1 in 0.2.20) echo 0.1.66;; 0.2.21) echo 0.1.67;; esac; }
RELS="0.2.20 0.2.21"
# the engines pick Level Zero V1 themselves; the selector keeps them off the OpenCL CPU device (same as the earlier B70 gates)
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
unset LD_LIBRARY_PATH OMP_PLACES OMP_PROC_BIND GOMP_CPU_AFFINITY KMP_AFFINITY

row() {  # <model> <release> <test> <value>
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$TSV"
  echo "RESULT $1 $2 | $3 | $4"
}
dmesg_count() { (dmesg 2>/dev/null | grep -ciE 'GPU HANG|xe.*(reset|timeout)|i915.*hang') || echo n/a; }
freegb() { df -BG --output=avail "$W" | tail -1 | tr -dc 0-9; }
exe_of() { echo "$PKG/$1-$2/$3"; }   # <release> kind exe: exe_of inkling 0.2.21 strata-inkling

do_pkgs() {
  echo "== host $(hostname) $(date -u +%FT%TZ)"; sycl-ls 2>/dev/null || true; df -h "$W" | tail -1; free -g | head -2
  echo "dmesg gpu hang/fault count before: $(dmesg_count)"
  local bad=0 rel kind tag ver f url sums want got d
  for rel in $RELS; do
    tag=$(tag_of "$rel"); ver=$(ver_of "$rel")
    curl -fsSL --retry 3 -o "$W/SHA256SUMS-$rel.txt" "$REL_URL/$tag/SHA256SUMS.txt" || { echo "no SHA256SUMS for $tag"; bad=1; continue; }
    for kind in inkling hy4 engine; do
      f=strata-$kind-$ver-linux-x86_64-sycl.tar.gz
      curl -fsSL --retry 3 --connect-timeout 30 -o "$W/$f" "$REL_URL/$tag/$f" || { echo "download failed $f"; bad=1; continue; }
      want=$(awk -v f="$f" '($2==f || $2=="*" f) {print $1}' "$W/SHA256SUMS-$rel.txt")
      got=$(sha256sum "$W/$f" | cut -d' ' -f1)
      if [ -n "$want" ] && [ "$want" = "$got" ]; then echo "sha256 OK  $f $got"; row pkg "$rel" "$kind sha256 vs SHA256SUMS.txt" OK
      else echo "sha256 BAD $f want=$want got=$got"; row pkg "$rel" "$kind sha256 vs SHA256SUMS.txt" "BAD want=$want got=$got"; bad=1; continue; fi
      d=$PKG/$kind-$rel; rm -rf "$d"; mkdir -p "$d"; tar xzf "$W/$f" -C "$d" || { echo "untar failed $f"; bad=1; continue; }
      python3 -I "$HERE/verify_component.py" "$d" || bad=1
      rm -f "$W/$f"
    done
  done
  for rel in $RELS; do
    echo "---- $rel ($(tag_of "$rel"))"
    for kind in inkling hy4 engine; do
      d=$PKG/$kind-$rel
      case $kind in inkling) x=strata-inkling;; hy4) x=strata-hy4;; engine) x=strata;; esac
      [ -x "$d/$x" ] || { echo "$d/$x missing"; bad=1; continue; }
      echo "$kind BUILD.json: $(cat "$d/BUILD.json")"
      echo "$kind ldd not found: $(ldd "$d/$x" 2>&1 | grep -c 'not found')"
      v=$("$d/$x" --version 2>&1 | head -3 | tr '\n' ' ')
      echo "$kind --version: $v"
      row "$kind" "$rel" "--version" "$v"
    done
  done
  echo "dmesg gpu hang/fault count after: $(dmesg_count)"
  return $bad
}

# --- Inkling -----------------------------------------------------------------------------------------------------------------------------
bench() {  # <label> <rel> <test> <model> <bench args...>   (draft off, one GPU)
  local L=$1 rel=$2 test=$3 M=$4; shift 4
  local exe line pf dc
  exe=$(exe_of inkling "$rel" strata-inkling)
  line=$(timeout 1500 "$exe" --bench --native "$M" --draft off --gpus 1 "$@" 2>"$LOGS/$L/err.$rel.$test.log" | grep -E '^\[bench\]' | head -1)
  echo "BENCH $L $rel $test | ${line:-FAILED}"
  if [ -z "$line" ]; then row "$L" "$rel" "$test prefill ppts" FAILED; tail -4 "$LOGS/$L/err.$rel.$test.log"; return 1; fi
  pf=$(echo "$line" | sed -n 's/.*prefill=\([0-9.]*\) ppts.*/\1/p'); dc=$(echo "$line" | sed -n 's/.*decode=\([0-9.]*\) tps agg.*/\1/p')
  row "$L" "$rel" "$test prefill ppts" "$pf"; row "$L" "$rel" "$test decode tok/s (total)" "$dc"
  grep -E "runner ready|VRAM experts" "$LOGS/$L/err.$rel.$test.log" | head -2 | cut -c1-260 | sed "s/^/INFO $L $rel $test | /"
}

greedy_ids() {  # <label> <rel> <model> <ids> <test-name> [extra engine args...]
  local L=$1 rel=$2 M=$3 ids=$4 name=$5; shift 5
  local exe out last
  exe=$(exe_of inkling "$rel" strata-inkling)
  out=$LOGS/$L/$name.$rel
  timeout 1800 "$exe" --native "$M" --gpus 1 --max-context 4096 "$@" --prompt-ids "$ids" --gen 64 > "$out.out" 2> "$out.err"
  local rc=$?
  last=$(grep -v '^[[:space:]]*$' "$out.out" | tail -n 1)
  if [ $rc -ne 0 ]; then
    row "$L" "$rel" "$name ids md5" "FAILED rc=$rc $(grep -iE 'error|missing|fail|unsupported|refus' "$out.err" | head -1 | cut -c1-200)"
    return 1
  fi
  row "$L" "$rel" "$name ids md5" "$(echo "$last" | md5sum | cut -c1-12)"
  row "$L" "$rel" "$name ids head" "$(echo "$last" | cut -c1-60)"
}

do_inkling() {
  local L=$1 URL=$2 F=$3 DURL=${4:-} DF=${5:-} SHA=${6:-}
  local M=$MODELS/$F DSP="" bad=0 rel r t exe I64
  mkdir -p "$LOGS/$L"
  echo "== $(date +%T) inkling $L: download"
  [ "$(freegb)" -ge 40 ] || { echo "not enough disk for $F"; return 1; }
  local t0; t0=$(date +%s)
  curl -fsSL --retry 3 --connect-timeout 30 --speed-limit 1000 --speed-time 60 -o "$M" "$URL" || { echo "download failed $F"; return 1; }
  echo "downloaded $F $(stat -c %s "$M") bytes in $(( $(date +%s) - t0 )) s"
  if [ -n "$SHA" ]; then echo "$SHA  $M" | sha256sum -c - || return 1; fi
  if [ -n "$DURL" ]; then
    DSP=$MODELS/$DF
    curl -fsSL --retry 3 --connect-timeout 30 --speed-limit 1000 --speed-time 60 -o "$DSP" "$DURL" || { echo "download failed $DF"; return 1; }
    ls -l "$DSP"
  fi
  echo "dmesg gpu hang/fault count before: $(dmesg_count)"
  I64=$(seq -s, 101 164)

  echo "== $L 1. selfcheck (.strata only)"
  if [ "${F##*.}" = "strata" ]; then
    for rel in $RELS; do
      exe=$(exe_of inkling "$rel" strata-inkling)
      timeout 900 "$exe" --selfcheck --native "$M" --gpus 1 --max-context 4096 > "$LOGS/$L/selfcheck.$rel.log" 2>&1
      r=$?; row "$L" "$rel" "selfcheck" "rc=$r $(grep -E 'selfcheck|SELFCHECK|PASS|FAIL' "$LOGS/$L/selfcheck.$rel.log" | head -1 | cut -c1-120)"
    done
  else
    echo "(GGUF: no selfcheck, same as the earlier gate)"
  fi

  echo "== $L 2. greedy: 64 prompt ids, 64 new tokens, --draft off, two runs per release, logits hash of 16 tokens"
  for rel in $RELS; do
    exe=$(exe_of inkling "$rel" strata-inkling)
    greedy_ids "$L" "$rel" "$M" "$I64" "greedy64 draft off run1" --draft off || bad=1
    greedy_ids "$L" "$rel" "$M" "$I64" "greedy64 draft off run2" --draft off || bad=1
    timeout 1800 "$exe" --native "$M" --gpus 1 --max-context 4096 --draft off --prompt-ids "$I64" --gen 16 --dump-ref "$LOGS/$L/l64.bin" > "$LOGS/$L/l64.$rel.log" 2>&1
    row "$L" "$rel" "logits64 sha256 (16 tokens)" "$(sha256sum "$LOGS/$L/l64.bin" 2>/dev/null | cut -c1-16)"
    rm -f "$LOGS/$L/l64.bin"
  done
  echo "== $L 3. greedy with the drafter (ids may differ by design)"
  for rel in $RELS; do
    if [ -n "$DSP" ]; then greedy_ids "$L" "$rel" "$M" "$I64" "greedy64 dspark gguf" --dspark "$DSP" --draft dspark || true
    else greedy_ids "$L" "$rel" "$M" "$I64" "greedy64 dspark embedded" --draft dspark || true; fi
  done

  echo "== $L 4. speed: prefill + decode, short (512) and 4K prompt, 2 rounds, release order flipped in round 2; C32 bench"
  for r in 1 2; do
    local order="0.2.20 0.2.21"; [ $r = 2 ] && order="0.2.21 0.2.20"
    for rel in $order; do bench "$L" "$rel" "short512" "$M" --parallel 1 --max-context 4096 --prompt-len 512 --gen 128 || bad=1; done
    for rel in $order; do bench "$L" "$rel" "4k" "$M" --parallel 1 --max-context 8192 --prompt-len 4096 --gen 128 || bad=1; done
    for rel in $order; do bench "$L" "$rel" "C32 512" "$M" --parallel 32 --max-context 4096 --prompt-len 512 --gen 64 || bad=1; done
  done
  echo "== $L 5. default environment (no UR_LOADER / selector set by us): short bench, one run per release"
  for rel in $RELS; do
    exe=$(exe_of inkling "$rel" strata-inkling)
    line=$(env -u ONEAPI_DEVICE_SELECTOR -u UR_LOADER_USE_LEVEL_ZERO_V2 timeout 1500 "$exe" --bench --native "$M" --draft off --gpus 1 --parallel 1 --max-context 4096 --prompt-len 512 --gen 128 2>"$LOGS/$L/err.$rel.bare.log" | grep -E '^\[bench\]' | head -1)
    bp=$(echo "$line" | sed -n 's/.*prefill=\([0-9.]*\) ppts.*/\1/p'); [ -n "$bp" ] || bp=FAILED
    row "$L" "$rel" "bare env short512 prefill ppts" "$bp"
  done

  echo "== $L 6. server on 0.2.21 (terminal tree $TERM_SRC): first chat, 4K needle, then one C32 burst"
  exe=$(exe_of inkling 0.2.21 strata-inkling)
  local DSA=(); [ -n "$DSP" ] && DSA=(--dspark "$DSP")
  "$W/venv/bin/python" "$ROOT/.github/ci/inkling/serve_chat.py" --out "$LOGS/$L/chat" --src "$TERM_SRC" --exe "$exe" --tok "$W/tok" --port 18781 --model-name "$L" --wait 1800 \
    --post "$ROOT/.github/ci/inkling/longctx_chat.py {port} 4096" -- --native "$M" ${DSA[@]+"${DSA[@]}"} --max-context 16384 --draft dspark 2>&1 | tee "$LOGS/$L/chat.summary"
  r=${PIPESTATUS[0]}; row "$L" 0.2.21 "server chat + 4K needle" "rc=$r $(grep -oE 'ANSWER-TEXT-[A-Z]+' "$LOGS/$L/chat.summary" | head -1) $(grep -oE 'needle4096 [0-9]+s.*(OK|WRONG)$' "$LOGS/$L/chat.summary" | grep -oE '(OK|WRONG)$' | head -1)"
  [ $r -eq 0 ] || bad=1
  "$W/venv/bin/python" "$ROOT/.github/ci/inkling/serve_chat.py" --out "$LOGS/$L/c32" --src "$TERM_SRC" --exe "$exe" --tok "$W/tok" --port 18782 --model-name "$L" --wait 1800 --no-chat \
    --post "$ROOT/.github/ci/inkling/c32_chat.py {port} 32 24" -- --native "$M" ${DSA[@]+"${DSA[@]}"} --max-context 4096 --parallel 32 --draft dspark 2>&1 | tee "$LOGS/$L/c32.summary"
  r=${PIPESTATUS[0]}; row "$L" 0.2.21 "server C32 burst" "rc=$r $(grep -oE 'c32@0: [0-9]+/[0-9]+ right[^|]*' "$LOGS/$L/c32.summary" | head -1 | cut -c1-110)"
  [ $r -eq 0 ] || bad=1

  echo "dmesg gpu hang/fault count after: $(dmesg_count)"
  rm -f "$M" "$DSP"
  echo "== $(date +%T) inkling $L done, failures: $bad"
  return $bad
}

do_setup_server() {  # terminal tree of v0.2.21 + venv + tokenizer files (once)
  [ -f "$W/.server-ready" ] && return 0
  TERM_SRC=$W/src0221; rm -rf "$TERM_SRC"; mkdir -p "$TERM_SRC"
  if (cd "$ROOT" && git fetch --no-tags --depth=1 origin d46f5f268f8ed797e83a5c63a69bffbd7a581cf6 >/dev/null 2>&1 && git archive FETCH_HEAD terminal | tar -x -C "$TERM_SRC"); then
    echo "terminal tree: v0.2.21 (d46f5f268f8e)"
  else
    echo "WARNING: could not fetch v0.2.21, using the terminal tree of this checkout"; rm -rf "$TERM_SRC"; TERM_SRC=$ROOT
  fi
  echo "$TERM_SRC" > "$W/.term-src"
  python3 -m venv "$W/venv" && "$W/venv/bin/pip" install -q --disable-pip-version-check -r "$TERM_SRC/terminal/requirements.txt" || return 1
  "$W/venv/bin/python" -c "import numpy, jinja2, regex, yaml; print('python deps ok')" || return 1
  mkdir -p "$W/tok"
  local f
  for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.jinja; do
    curl -fsSL --retry 3 -o "$W/tok/$f" "$HF_STRATA/$f" || return 1
  done
  touch "$W/.server-ready"
}

do_hy4() {
  echo "== hy4"
  [ "$(freegb)" -ge 68 ] || { echo "not enough disk for the 62 GB Hy4-Flash-SQ-64.strata"; return 1; }
  local M=$MODELS/Hy4-Flash-SQ-64.strata t0; t0=$(date +%s)
  curl -fsSL --retry 2 --connect-timeout 30 --speed-limit 1000 --speed-time 60 -o "$M" "$HF_HY4/Hy4-Flash-SQ-64.strata" || { echo "download failed"; return 1; }
  echo "downloaded Hy4-Flash-SQ-64.strata $(stat -c %s "$M") bytes in $(( $(date +%s) - t0 )) s"
  mkdir -p "$LOGS/hy4" "$W/hy4tok"
  local b20 b21
  b20=$(exe_of hy4 0.2.20 strata-hy4); b21=$(exe_of hy4 0.2.21 strata-hy4)
  "$b21" --export-tokenizer "$M" "$W/hy4tok" > "$LOGS/hy4/export-tokenizer.log" 2>&1 || { echo "export-tokenizer failed"; tail -5 "$LOGS/hy4/export-tokenizer.log"; rm -f "$M"; return 1; }
  echo "dmesg gpu hang/fault count before: $(dmesg_count)"
  python3 -I "$HERE/hy4_rel.py" "$M" "$W/hy4tok" "$b20" "$b21" "$LOGS/hy4" "$TSV" 2>&1 | tee "$LOGS/hy4.summary"
  local rc=${PIPESTATUS[0]}
  echo "dmesg gpu hang/fault count after: $(dmesg_count)"
  rm -f "$M"
  return $rc
}

TERM_SRC=$(cat "$W/.term-src" 2>/dev/null || echo "$ROOT")
cmd=${1:-}; shift || true
case $cmd in
  pkgs) do_pkgs; rc=$?; do_setup_server || rc=1; exit $rc;;
  inkling) TERM_SRC=$(cat "$W/.term-src" 2>/dev/null || echo "$ROOT"); do_inkling "$@";;
  hy4) do_hy4;;
  summary) python3 -I "$HERE/summarize.py" "$TSV" "$LOGS/summary.md";;
  *) echo "usage: $0 pkgs|inkling|hy4|summary"; exit 2;;
esac
