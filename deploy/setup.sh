#!/usr/bin/env bash
# One-time bootstrap for a fresh Ubuntu 24.04 server (run as root):
#   curl -fsSL https://raw.githubusercontent.com/JeremyUtomo/worlds-2026-predictor/main/deploy/setup.sh | sudo bash
# Point the DNS A record for worlds.jeremyutomo.com at this server first, so Caddy
# can get the HTTPS certificate on the first start.
set -euxo pipefail

REPO_URL="${REPO_URL:-https://github.com/JeremyUtomo/worlds-2026-predictor.git}"
APP_DIR=/opt/worlds-2026-predictor
DATA_DIR=/var/lib/worlds

# swap: the model build briefly needs more memory than a small instance has
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

apt-get update
apt-get install -y git jq curl docker.io docker-compose-v2
systemctl enable --now docker

mkdir -p "$DATA_DIR"
chown 1000:1000 "$DATA_DIR"   # the container user

if [ ! -d "$APP_DIR/.git" ]; then
  git clone "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"
docker compose up -d --build

# pull-based auto-deploy: every 2 minutes, deploy main once CI has passed
cat > /etc/systemd/system/worlds-deploy.service <<EOF
[Unit]
Description=Deploy worlds-2026-predictor if main has a new commit with green CI
After=docker.service network-online.target

[Service]
Type=oneshot
WorkingDirectory=$APP_DIR
ExecStart=/bin/bash $APP_DIR/deploy/auto-deploy.sh
EOF

cat > /etc/systemd/system/worlds-deploy.timer <<'EOF'
[Unit]
Description=Check for worlds-2026-predictor updates every 2 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=2min

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now worlds-deploy.timer
echo "Done. Site: https://worlds.jeremyutomo.com  (logs: docker compose -f $APP_DIR/docker-compose.yml logs -f)"
