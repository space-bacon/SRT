#!/bin/sh
# Keep the lab tunnel alive. lab.sunstonenorth.com serves a static frontend from
# the Linode front, but every /api/* call is proxied back through WireGuard to
# the machine running the model. If wg0 drops, the site loads and the demo
# hangs, which is the failure that is easy to miss.
#
# Checked two ways on purpose: the interface can exist while the peer is
# unreachable (common after sleep or a network change), and that state looks
# healthy to anything that only tests for the interface.
set -u

PEER=10.77.0.1
IFACE=wg0
WG_QUICK=/opt/homebrew/bin/wg-quick

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S') wg-keepalive: $*"; }

# Route presence, not `wg show`: `wg show` needs root, so a non-root run of this
# script would report a healthy tunnel as down and bounce it.
if ! /usr/sbin/netstat -rn | grep -q "^${PEER}"; then
    log "$IFACE is not up, starting"
    "$WG_QUICK" up "$IFACE" && log "$IFACE up" || log "failed to bring $IFACE up"
    exit 0
fi

if ! /sbin/ping -c 3 -t 5 "$PEER" >/dev/null 2>&1; then
    log "$IFACE exists but $PEER is unreachable, bouncing"
    "$WG_QUICK" down "$IFACE" >/dev/null 2>&1
    "$WG_QUICK" up "$IFACE" && log "$IFACE restarted" || log "failed to restart $IFACE"
fi
