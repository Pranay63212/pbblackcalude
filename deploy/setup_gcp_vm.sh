#!/usr/bin/env bash
# One-time setup on a fresh Google Cloud VM (Ubuntu 24.04 LTS or Debian 12).
#
#   git clone https://github.com/Pranay63212/pbblackcalude
#   bash pbblackcalude/deploy/setup_gcp_vm.sh
#
# What it does:
#   1. installs Python + git, creates a virtual environment, installs requirements
#   2. runs `python tests.py` (stops if any test fails)
#   3. asks for your Alpaca PAPER keys and ntfy address and saves them in
#      ~/.pbblake.env (readable only by you; never inside the repository)
#   4. sends a test phone alert
#   5. installs a systemd service that starts the scanner now, after every
#      reboot, and again 30 s after any crash
# Re-run with --reset-keys to enter new keys. Paper trading only: --live is refused.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$HOME/.pbblake.env"
SERVICE=pbblake-scanner
DEFAULT_FLAGS="--feed alpaca --trade --push-summary --daily-limits relaxed --session extended --news-filter --pair-filter"
PY="$REPO_DIR/.venv/bin/python"

say() { printf '\n==> %s\n' "$*"; }

say "Installing system packages"
sudo apt-get update -y -qq
sudo apt-get install -y -qq python3 python3-venv python3-pip git >/dev/null

if ! swapon --show | grep -q .; then
  say "Adding 1 GB of swap (safety margin on the 1 GB free-tier machine)"
  sudo fallocate -l 1G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile && echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

say "Creating the Python environment and installing requirements"
python3 -m venv "$REPO_DIR/.venv"
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q -r "$REPO_DIR/requirements.txt"

say "Running the test suite (must pass)"
(cd "$REPO_DIR" && "$PY" tests.py 2>&1 | tail -3)
(cd "$REPO_DIR" && "$PY" tests.py >/dev/null 2>&1) || { echo "TESTS FAILED - not installing the service."; exit 1; }

if [[ ! -f "$ENV_FILE" || "${1:-}" == "--reset-keys" ]]; then
  say "Your keys (typed here, saved only on this server in $ENV_FILE)"
  read -rp  "Alpaca PAPER API key ID: " KEY_ID
  read -rsp "Alpaca PAPER secret key (hidden while typing): " SECRET; echo
  read -rp  "ntfy address, e.g. https://ntfy.sh/your-channel: " HOOK
  [[ "$HOOK" == https://ntfy.sh/* ]] || echo "WARNING: that address does not start with https://ntfy.sh/ - check the spelling."
  for v in "$KEY_ID" "$SECRET" "$HOOK"; do
    if [[ -z "$v" || "$v" =~ [[:space:]\'\"\\\$] ]]; then
      echo "A value was empty or contained spaces/quotes - please run the setup again and paste carefully."; exit 1
    fi
  done
  umask 077
  # single-quoted: read correctly both by bash (`. file`) and by systemd's EnvironmentFile
  cat > "$ENV_FILE" <<EOF
APCA_API_KEY_ID='$KEY_ID'
APCA_API_SECRET_KEY='$SECRET'
FBT_ALERT_WEBHOOK='$HOOK'
FBT_FLAGS='$DEFAULT_FLAGS'
EOF
  chmod 600 "$ENV_FILE"
fi
if grep -q -- "--live" "$ENV_FILE"; then
  echo "Refusing: FBT_FLAGS contains --live. This project only paper trades."; exit 1
fi

say "Sending a test alert to your phone"
(set -a; . "$ENV_FILE"; set +a; cd "$REPO_DIR" && "$PY" -m fbt.scanner test-alert) || \
  echo "Phone alert failed (see the message above). The scanner will still run; fix FBT_ALERT_WEBHOOK with --reset-keys."

say "Installing the always-on service"
sudo tee /etc/systemd/system/$SERVICE.service >/dev/null <<EOF
[Unit]
Description=PB Blake strategy scanner (Alpaca PAPER trading)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$REPO_DIR
EnvironmentFile=$ENV_FILE
Environment=PYTHONUNBUFFERED=1
ExecStart=$PY -m fbt.scanner live \$FBT_FLAGS
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now $SERVICE >/dev/null
sleep 5
sudo systemctl --no-pager --lines=8 status $SERVICE || true

say "Done. The scanner runs on its own now."
echo "  Watch it live:   journalctl -u $SERVICE -f        (Ctrl+C stops watching, not the scanner)"
echo "  Update later:    bash $REPO_DIR/deploy/update.sh"
echo "  Your trades:     cat $REPO_DIR/results/paper_trades.csv"
