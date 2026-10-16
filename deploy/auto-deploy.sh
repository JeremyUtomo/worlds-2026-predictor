#!/usr/bin/env bash
# Pull-based deploy, run every 2 minutes by worlds-deploy.timer.
# Deploys origin/main only after every GitHub check on that commit has passed,
# so the server needs no SSH keys or cloud credentials. A commit whose CI failed
# is remembered and never retried.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${REPO:-JeremyUtomo/worlds-2026-predictor}"
STATE_DIR=/var/lib/worlds-deploy
SKIP_FILE="$STATE_DIR/failed-commits"
mkdir -p "$STATE_DIR"
touch "$SKIP_FILE"

cd "$APP_DIR"
git fetch --quiet origin main
target="$(git rev-parse origin/main)"
current="$(git rev-parse HEAD)"

if [ "$target" = "$current" ]; then
  exit 0
fi
if grep -qx "$target" "$SKIP_FILE"; then
  exit 0
fi

checks="$(curl -fsSL -H 'Accept: application/vnd.github+json' \
  "https://api.github.com/repos/$REPO/commits/$target/check-runs")"
total="$(jq '.check_runs | length' <<<"$checks")"
pending="$(jq '[.check_runs[] | select(.status != "completed")] | length' <<<"$checks")"
failed="$(jq '[.check_runs[] | select(.status == "completed")
               | select(.conclusion != "success" and .conclusion != "skipped" and .conclusion != "neutral")]
              | length' <<<"$checks")"

if [ "$total" -eq 0 ] || [ "$pending" -gt 0 ]; then
  echo "waiting for CI on ${target:0:7} ($pending of $total checks pending)"
  exit 0
fi
if [ "$failed" -gt 0 ]; then
  echo "CI failed on ${target:0:7}; not deploying it"
  echo "$target" >> "$SKIP_FILE"
  exit 0
fi

echo "deploying ${target:0:7} (was ${current:0:7})"
git reset --hard "$target"
docker compose up -d --build --remove-orphans
docker image prune -f
