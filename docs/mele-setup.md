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

The Mele logs in with the owner's Google account, so dad needs no Tailscale account. He powers it on and plugs in Ethernet.

To give him access too, in the Tailscale admin console go to Machines -> `mele` -> Share and send him the link. He signs in with his own account. Don't share the Google login.

Power off: a short press of the power button shuts down cleanly (logind `HandlePowerKey=poweroff`). Wait for the light to go off before unplugging.

## Before the Florida trip

- [x] Mele: key expiry disabled, service runs as `astro` from `/home/astro/astro`.
- [ ] Mele BIOS: power on after AC loss, so plugging it in boots it.
- [ ] Tablet: Tailscale app (owner's Google login), key expiry disabled, Android Always-on VPN,
      app added to the home screen, mic and location allowed.
- [ ] Tablet: Chrome Remote Desktop (or similar), so the developer can see the tablet's screen when
      debugging remotely.
- [ ] Offline test: tablet and Mele on a phone hotspot with mobile data off; does the app load
      and does voice work? (see dev-runbook, "Field use")
- [ ] Full sim session on the Mele through the tablet: voice -> plan -> guide -> capture -> image.
- [ ] Dark frames from the main camera, capped with foil (the plastic cap passes IR).
- [ ] Power-bank test: both cameras + stacking for 1 h, no reboot when plugging ports.
- [ ] MCU: BME280 wired and reporting `ENV`; heater failsafe; quadrature test with jumpers.
- [ ] Pack: multimeter, logic analyzer (encoder RJ pinout), USB-RS232 for the handset fallback.

## Remote notes (from the first bring-up)

- The dev box runs Tailscale in userspace mode: plain `ssh mele` fails, use `tailscale ssh astro@mele`. The user is the Mele's Linux account, not the Tailscale login; "tailnet policy does not permit you to SSH as user X" usually means account X doesn't exist on the Mele.
- Field WiFi lives in `/etc/netplan/60-wifi.yaml` (mode 600): the Pixel hotspot, used when Ethernet is unplugged. Add networks there while Ethernet is connected. Both links are `optional` so boot doesn't wait for one.
- `astro` needs `video` (cameras), `dialout` (MCU serial) and the SDK in `/home/astro/sdk`.
- Optional: passwordless sudo for that user (`/etc/sudoers.d/<user>`) lets Claude run installs unattended, but it means any session on that account has root. The Mele is reachable only over your tailnet; drop the file when the bring-up is done.
- `scripts/setup.sh --minipc` needs HTTPS certificates enabled in the Tailscale admin console (DNS), or it stops at the cert step.
- The closed-source SVBony SDK is not in the repo: copy it with `tar -C ~ -cz sdk | tailscale ssh <user>@mele 'tar -C ~ -xz'`.
- The classic Nano (CH340) appears as `/dev/ttyUSB0`, not `ttyACM*`. It runs `astro_mcu` (flashed with `avrdude`, old bootloader at 57600; see `firmware/README.md`). BME280, encoders and heaters are not wired yet.
