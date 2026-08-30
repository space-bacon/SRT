#!/bin/bash
# Pull each result off the rented box the moment it lands, then back it up twice.
#
# Waiting for the whole chain before saving anything is how we lost v12 and an
# in-progress v13b: vast.ai reclaims hosts without warning, and a file that
# exists only on a rented box is one reclaim away from gone. So this watches for
# each artifact individually and pulls it as soon as it appears, rather than
# gating on the final marker.
#
# Small JSON goes into git. Large .npz is gitignored, so it goes to Supabase and
# to the LaCie, which are the two places that survive both a dead box and a dead
# laptop.
#
# rsync rather than scp for --partial, so a dropped connection costs the
# remainder of a file instead of the whole thing, and this box has been dropping
# ssh under load all night. NOTE: macOS ships openrsync ("2.6.9 compatible"),
# which has --partial/--append/--inplace but NOT --append-verify. That flag is
# rsync 3.0+ and fails outright here.
#
#   nohup bash scripts/bank_results.sh > logs/bank.log 2>&1 &
set -u
SSH_OPTS="-o ServerAliveInterval=20 -o ServerAliveCountMax=100 -p 27357"
BOX=root@ssh4.vast.ai
REPO=/Users/burtron/development/srt-adapter
STATES=$REPO/artifacts/nla/omni/states
NLA=$REPO/artifacts/nla
LACIE=/Volumes/LaCie/SRT-ARCHIVE
FINAL=/root/decode_states.npz
DEADLINE=$(( $(date +%s) + 6*3600 ))

JSONS="cxr14_probe_aria.json cxr14_probe_mistral.json cxr14_probe_qwen3omni.json
       cxr14_vendor_compare.json cxr14_transport.json cxr14_ensemble4.json
       decode_states_captions.json"
NPZS="cxr14_aria.npz cxr14_mistral.npz cxr14_qwen3omni.npz decode_states.npz"

mkdir -p "$STATES" "$NLA" "$REPO/logs"
cd "$REPO" || exit 1
say () { echo "[$(date '+%H:%M:%S')] $*"; }

pull () {  # remote_name  local_dir -> 0 only when a NEW file actually arrived
  local f=$1 d=$2
  [ -s "$d/$f" ] && return 1
  ssh $SSH_OPTS $BOX "test -s /root/$f" 2>/dev/null || return 1
  # Size must be stable across two reads, or we would copy a file mid-write.
  local a b
  a=$(ssh $SSH_OPTS $BOX "stat -c %s /root/$f" 2>/dev/null)
  sleep 20
  b=$(ssh $SSH_OPTS $BOX "stat -c %s /root/$f" 2>/dev/null)
  [ "$a" != "$b" ] && { say "still growing, will retry: $f"; return 1; }
  say "pulling $f ($(echo "$a" | awk '{printf "%.1f MB", $1/1e6}'))"
  if rsync -e "ssh $SSH_OPTS" --partial -q "$BOX:/root/$f" "$d/$f"; then
    say "  got $f"
    return 0
  fi
  say "  FAILED $f"
  return 1
}

say "watching the box; will bank each artifact as it lands"
while :; do
  fresh=""
  for f in $JSONS; do pull "$f" "$NLA" && fresh="$fresh $f"; done
  for f in $NPZS;  do pull "$f" "$STATES"; done

  # Stage only what this run pulled. A blanket add on artifacts/nla/*.json
  # sweeps up unrelated local work under a commit message about the box.
  if [ -n "$fresh" ]; then
    staged=0
    for f in $fresh; do
      [ -s "$NLA/$f" ] && git add "$NLA/$f" && staged=1
    done
    if [ "$staged" = 1 ]; then
      git commit -q -m "bank from the box:$fresh" -m \
"Pulled automatically as it landed rather than at the end of the chain. A file
that exists only on a rented vast.ai box is one host-reclaim away from gone,
which has already cost us two runs." && git push -q origin main \
        && say "committed +$fresh"
    fi
  fi

  if ssh $SSH_OPTS $BOX "test -s $FINAL" 2>/dev/null && [ -s "$STATES/decode_states.npz" ]; then
    say "final marker present and pulled; moving to backup"
    break
  fi
  [ "$(date +%s)" -gt "$DEADLINE" ] && { say "deadline reached, backing up what we have"; break; }
  sleep 180
done

say "=== supabase"
have=""
for f in $NPZS cxr14_gemma4.npz nlst_gemma4.npz; do
  [ -s "$STATES/$f" ] && have="$have $STATES/$f"
done
# shellcheck disable=SC2086
python scripts/supabase_backup.py --prefix cxr14 $have && say "supabase done" \
  || say "SUPABASE FAILED"
python scripts/supabase_backup.py --prefix cxr14_results artifacts/nla/*.json \
  && say "supabase json done" || say "SUPABASE JSON FAILED"

say "=== lacie"
if [ -d /Volumes/LaCie ]; then
  python scripts/archive_to_drive.py --dest "$LACIE" && say "lacie done" \
    || say "LACIE FAILED"
else
  say "LACIE NOT MOUNTED, skipped"
fi
say "=== all banking complete"
