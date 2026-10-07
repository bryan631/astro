# Troubleshooting

## First: run the check

From the dev machine (on the tailnet):

```bash
scripts/check.sh mele
```

It pings the Mele over Tailscale, loads the web page, then runs the same script on the Mele over
SSH. On the Mele itself (or if SSH is the problem but you have a keyboard there), run
`~/astro/scripts/check.sh`. Each line says `ok`, `warn` or `FAIL` with a hint; the exit code is the
number of failures.

What it checks on the Mele:

| Section | Check | Expect |
|---|---|---|
| network | default route, WiFi name, internet, Tailscale | a route; `NETGEAR30` or the Pixel hotspot (or Ethernet) |
| devices | SVBony cameras (`lsusb -d f266:`), MCU serial port, `astro` groups | 2 cameras; `/dev/ttyUSB0` or `ttyACM0` for the MCU; `video`, `dialout` |
| service | `astro` service, web on :8443, errors in the last 15 min, cert expiry | active, 200, none, > 7 days |
| system | free disk, load | > 2 GB |

## Symptoms

**`tailscale ssh` prints "Tailscale SSH requires an additional check" and a link.** The SSH policy
re-checks the login periodically. Open the link, approve, run again. The web UI keeps working
meanwhile.

**Tailnet ping fails.** The Mele is off, has no network, or Tailscale is down on either side.
Ask dad: is the Mele's light on? Power-cycle it (short press of the power button, wait for the light
to go off, press again). It joins `NETGEAR30` by itself; Ethernet works too. On the dev machine,
`tailscale status` must list it. On the Chromebook, Tailscale runs in userspace mode: plain
`ssh mele` fails, use `tailscale ssh astro@mele`.

**Web page doesn't load on the phone or tablet.** Use the full
`https://mele.taila252f9.ts.net:8443/` (with `https://`); the raw Tailscale IP gives a certificate
warning and breaks the mic. The phone or tablet must be on the tailnet (Tailscale app on).

**Page loads but is old.** A deploy reloads open pages by itself. If not, reload once; incognito
shows the current build.

**"The telescope isn't ready: ..."** The server couldn't start the hardware; the text says what.
Usually a camera not found: check the cables, then `lsusb -d f266:` (want 2 lines) and
`sudo systemctl restart astro`.

**MCU light is on but no serial port.** One of the Mele's USB-C ports is power-only: power goes
there, the MCU on the other. A charge-only cable does the same. The classic Nano shows up as
`/dev/ttyUSB0`.

**Voice: "I can't use the microphone".** The page must be opened over `https://` with the ts.net
name, and the mic allowed for the site. Hands-free needs `scripts/install-vad.sh` run once.

**Service failing.** `journalctl -u astro -n 100 --no-pager` shows why;
`sudo systemctl restart astro` restarts it. Deploy: `cd ~/astro && git pull && sudo systemctl
restart astro`.

See also: [mele-setup.md](mele-setup.md) (WiFi, Tailscale, cabling), [dev-runbook.md](dev-runbook.md).
