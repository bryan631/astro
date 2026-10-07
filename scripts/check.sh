#!/usr/bin/env bash
# Connection check for remote troubleshooting (docs/troubleshooting.md).
#   On the Mele:           scripts/check.sh
#   From the dev machine:  scripts/check.sh mele   (tailnet + web, then the Mele's own checks over SSH)
# Prints one line per check: ok / FAIL / warn, with a hint. Exit code = number of failures.
set -uo pipefail
PORT=${ASTRO_PORT:-8443}
fails=0
ok()   { printf '  ok    %s\n' "$*"; }
fail() { printf '  FAIL  %s\n' "$*"; fails=$((fails + 1)); }
warn() { printf '  warn  %s\n' "$*"; }
t() { timeout 10 "$@" 2>/dev/null; }  # nothing here may hang the check

remote() {  # from the dev machine: can we reach the Mele at all?
  local host=$1
  echo "== $host from $(hostname)"
  if t tailscale ping -c 1 "$host" >/dev/null; then ok "tailnet ping $host"
  else fail "tailnet ping $host: Mele off, no network, or Tailscale down on either side"; fi
  local name; name=$(t tailscale status --json | sed -n "s/.*\"DNSName\": *\"\($host\.[^\"]*\)\.\".*/\1/p" | head -1)
  local code; code=$(t curl -sk -o /dev/null -w '%{http_code}' "https://${name:-$host}:$PORT/")
  [ "$code" = 200 ] && ok "web https://$name:$PORT/" || fail "web https://${name:-$host}:$PORT/ answered '${code:-nothing}': service down?"
  echo "== on $host (if SSH prints a login link, open it: this waits 5 min)"
  timeout 300 tailscale ssh "astro@$host" 'bash -s' < "$0"
  local rc=$?
  if [ $rc -eq 124 ] || [ $rc -eq 255 ]; then fail "ssh astro@$host (exit $rc)"; else fails=$((fails + rc)); fi
}

local_checks() {
  echo "== network"
  local dev; dev=$(ip route show default 2>/dev/null | awk '{print $5; exit}')
  [ -n "$dev" ] && ok "default route via $dev" || fail "no default route: no Ethernet or WiFi"
  local ssid; ssid=$(t iw dev | awk '/ssid/ {print $2; exit}')
  [ -n "$ssid" ] && ok "WiFi: $ssid" || warn "no WiFi link (fine on Ethernet)"
  t ping -c 1 -W 3 1.1.1.1 >/dev/null && ok "internet (ping 1.1.1.1)" || warn "no internet: field mode works offline, but not Tailscale from outside"
  t tailscale status --self --peers=false >/dev/null && ok "tailscale up ($(t tailscale ip -4))" || fail "tailscale down: sudo tailscale up --ssh"

  echo "== devices"
  local cams; cams=$(lsusb -d f266: 2>/dev/null | wc -l)
  [ "$cams" -ge 2 ] && ok "$cams SVBony cameras" || fail "$cams SVBony cameras (want 2): check USB cables; lsusb"
  local ports; ports=$(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | tr '\n' ' ')
  [ -n "$ports" ] && ok "serial: $ports" || warn "no MCU serial port: data port, not the power-only USB-C; charge-only cable? (needed for mount driver mcu)"
  local groups; groups=$(id -nG astro 2>/dev/null)
  [[ " $groups " == *" video "* && " $groups " == *" dialout "* ]] && ok "astro in video, dialout" || fail "astro groups: $groups (want video, dialout)"

  echo "== service"
  [ "$(t systemctl is-active astro)" = active ] && ok "astro service active" || fail "astro service not active: journalctl -u astro -n 50"
  local code; code=$(t curl -sk -o /dev/null -w '%{http_code}' "https://localhost:$PORT/")
  [ "$code" = 200 ] && ok "web on :$PORT" || fail "web on :$PORT answered '${code:-nothing}'"
  local errs; errs=$(t journalctl -u astro --since "-15 min" --no-pager | grep -ciE 'error|traceback')
  [ "${errs:-0}" -eq 0 ] && ok "no errors in the last 15 min of logs" || warn "$errs error lines in 15 min: journalctl -u astro --since -15min"
  local cert; cert=$(ls ~astro/astro/certs/*.crt 2>/dev/null | head -1)
  if [ -n "$cert" ]; then
    openssl x509 -checkend 604800 -noout -in "$cert" >/dev/null && ok "cert valid > 7 days" || warn "cert expires within 7 days: scripts/tailscale-cert.sh"
  else warn "no cert: plain HTTP, the tablet mic won't work"; fi

  echo "== system"
  local free; free=$(df -P ~astro | awk 'NR==2 {print int($4/1048576)}')
  [ "${free:-0}" -ge 2 ] && ok "disk: ${free} GB free" || fail "disk: ${free} GB free: prune data/"
  ok "load: $(cut -d' ' -f1-3 /proc/loadavg), $(uptime -p 2>/dev/null)"
}

if [ $# -gt 0 ]; then remote "$1"; else local_checks; fi
echo "== $fails failure(s)"
exit "$fails"
