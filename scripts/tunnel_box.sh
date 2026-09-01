#!/bin/bash
# Self-healing SSH tunnel to the consensus demo box.
# The box closes connections every minute or two, so a bare `ssh -L` is not enough.
#   usage: scripts/tunnel_box.sh [local_port] [remote_port]
set -u

LPORT="${1:-8081}"
RPORT="${2:-8081}"
BOX_PORT=53554
BOX=root@154.64.230.50

echo "tunnel localhost:${LPORT} -> box:${RPORT}   (Ctrl-C to stop)"

n=0
while true; do
  n=$((n + 1))
  ssh -N -L "${LPORT}:localhost:${RPORT}" \
      -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
      -o TCPKeepAlive=yes -o ExitOnForwardFailure=yes \
      -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new \
      -p "${BOX_PORT}" "${BOX}" 2>&1 | grep -vE "vast\.ai|Have fun|AI agents:|^$"
  echo "$(date '+%H:%M:%S')  link #${n} dropped, reconnecting in 3s..."
  sleep 3
done
