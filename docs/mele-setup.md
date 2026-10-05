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

## Remote notes (from the first bring-up)

- The dev box runs Tailscale in userspace mode: plain `ssh mele` fails, use `tailscale ssh <user>@mele`. The user is the Mele's Linux account, not the Tailscale login.
- Optional: passwordless sudo for that user (`/etc/sudoers.d/<user>`) lets Claude run installs unattended, but it means any session on that account has root. The Mele is reachable only over your tailnet; drop the file when the bring-up is done.
- `scripts/setup.sh --minipc` needs HTTPS certificates enabled in the Tailscale admin console (DNS), or it stops at the cert step.
- The closed-source SVBony SDK is not in the repo: copy it with `tar -C ~ -cz sdk | tailscale ssh <user>@mele 'tar -C ~ -xz'`.
- The classic Nano (CH340) appears as `/dev/ttyUSB0`, not `ttyACM*`; the firmware port to it is on its own branch.
