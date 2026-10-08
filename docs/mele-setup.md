# Mele setup (SSH + Tailscale)

On the Mele (Ubuntu Server), run:

```bash
sudo apt update && sudo apt install -y curl openssh-server avahi-daemon tmux git \
&& sudo systemctl enable --now ssh \
&& sudo hostnamectl set-hostname mele \
&& curl -fsSL https://tailscale.com/install.sh | sh \
&& sudo tailscale up --ssh
```

Open the printed login URL and approve the device. Install Tailscale on the dev machine too, then:

```bash
ssh <mele-user>@mele
```

Tailscale SSH needs no keys. In the Tailscale admin console, disable key expiry for the Mele.

## Florida (dad's place)

The Mele logs in with the owner's Google account, so dad needs no Tailscale account. He just powers it on: it joins his WiFi (`NETGEAR30`) by itself. Ethernet works too.

Cables: one of the Mele's USB-C ports is power-only. Power goes there; the MCU goes on the other USB-C (data) port. If the MCU's light is on but it doesn't show up as a serial port, it's on the power-only port or on a charge-only cable.

To give him access too, in the Tailscale admin console go to Machines -> `mele` -> Share and send him the link. He signs in with his own account. Don't share the Google login.

Power off: a short press of the power button shuts down cleanly (logind `HandlePowerKey=poweroff`). Wait for the light to go off before unplugging.

## Before the Florida trip

- [x] Mele: key expiry disabled, service runs as `astro` from `/home/astro/astro`.
- [ ] Mele BIOS: power on after AC loss, so plugging it in boots it.
- [ ] Tablet: Tailscale app (owner's Google login), key expiry disabled, Android Always-on VPN,
      app added to the home screen, mic and location allowed.
- [ ] Tablet: a way for the developer to see its screen when debugging remotely, e.g. `scrcpy` over
      adb wireless debugging across the tailnet. Verify it before the trip.
- [ ] Offline test: tablet and Mele on a phone hotspot with mobile data off; does the app load
      and does voice work? (see dev-runbook, "Field use")
- [ ] Full sim session on the Mele through the tablet: voice -> plan -> guide -> capture -> image.
- [ ] Dark frames from the main camera, capped with foil (the plastic cap passes IR).
- [ ] Power-bank test: both cameras + stacking for 1 h, no reboot when plugging ports.
- [ ] MCU: BME280 wired and reporting `ENV`; heater failsafe; quadrature test with jumpers.
- [ ] Pack: multimeter, logic analyzer (encoder RJ pinout), USB-RS232 for the handset fallback.

## Remote notes (from the first bring-up)

- The dev box runs Tailscale in userspace mode: plain `ssh mele` fails, use `tailscale ssh astro@mele`. The user is the Mele's Linux account, not the Tailscale login; "tailnet policy does not permit you to SSH as user X" means either the Tailscale SSH policy doesn't allow login as X or account X doesn't exist on the Mele; check both.
- Field WiFi lives in `/etc/netplan/60-wifi.yaml` (mode 600): the Pixel hotspot, used when Ethernet is unplugged. Add networks there while Ethernet is connected. Both links are `optional` so boot doesn't wait for one.
- Open the web UI with the full `https://mele.<tailnet>.ts.net:8443/` (Chrome drops the scheme otherwise). The cert covers the name only, so the raw Tailscale IP shows a certificate warning and breaks the mic.
- Field WiFi so far: Pixel hotspot, plus dad's `netgear30` (2.4 GHz only) and `NETGEAR30` (2.4 and 5 GHz). Prefer `NETGEAR30`; dropping the `netgear30` entry is pending.
- `astro` needs `video` (cameras), `dialout` (MCU serial) and the SDK in `/home/astro/sdk`.
- Optional: passwordless sudo for that user (`/etc/sudoers.d/<user>`) lets Claude run installs unattended, but it means any session on that account has root. The Mele is reachable only over your tailnet; drop the file when the bring-up is done.
- `scripts/setup.sh --minipc` needs HTTPS certificates enabled in the Tailscale admin console (DNS), or it stops at the cert step.
- The closed-source SVBony SDK is not in the repo: copy it with `tar -C ~ -cz sdk | tailscale ssh <user>@mele 'tar -C ~ -xz'`.
- The classic Nano (CH340) appears as `/dev/ttyUSB0`, not `ttyACM*`. It runs `astro_mcu` (flashed with `avrdude`, old bootloader at 57600; see `firmware/README.md`). Encoders are wired and verified (see plan.md); BME280 and heaters are not wired yet.
