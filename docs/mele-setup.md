# Mele setup (SSH + Tailscale)

On the Mele (Ubuntu Server), run:

```bash
sudo apt update && sudo apt install -y openssh-server avahi-daemon tmux git \
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
