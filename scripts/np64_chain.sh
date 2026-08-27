#!/bin/bash
# Score np64 the moment training saves its final checkpoint.
#
# Gates on the trainer's own "(final)" line rather than pgrep: a pgrep pattern
# also matches the ssh launcher wrapper that carries the pattern in its own
# cmdline, which has hung this chain twice before.
set -u
LOG=/root/np64.log
until grep -q "(final)" "$LOG" 2>/dev/null; do sleep 30; done
sleep 10

cd /root/srt-adapter
/venv/main/bin/python scripts/eval_sunstone_verbalizer.py \
  --ckpt /root/reader_all5_np64.pt \
  --vecs /root/gal_all5.npy \
  --caps /root/full_caps.json \
  --head-path /root/head_all5.pt \
  --n 500 \
  --out /root/verb_eval_all5_np64.json
