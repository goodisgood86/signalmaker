#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_SCRIPT="$ROOT_DIR/scripts/pass_check_weekly_backup.sh"
BATCH_SCRIPT="$ROOT_DIR/scripts/pass_check_weekly_batch.sh"

AGENT_DIR="$HOME/Library/LaunchAgents"
mkdir -p "$AGENT_DIR"

BACKUP_PLIST="$AGENT_DIR/com.coin.passcheck.backup.plist"
BATCH_PLIST="$AGENT_DIR/com.coin.passcheck.batch.plist"

cat > "$BACKUP_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.coin.passcheck.backup</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>$BACKUP_SCRIPT</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Weekday</key><integer>1</integer>
    <key>Hour</key><integer>2</integer>
    <key>Minute</key><integer>50</integer>
  </dict>
  <key>StandardOutPath</key>
  <string>/tmp/com.coin.passcheck.backup.out.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/com.coin.passcheck.backup.err.log</string>
  <key>RunAtLoad</key>
  <false/>
</dict>
</plist>
PLIST

cat > "$BATCH_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.coin.passcheck.batch</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>$BATCH_SCRIPT</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Weekday</key><integer>1</integer>
    <key>Hour</key><integer>3</integer>
    <key>Minute</key><integer>10</integer>
  </dict>
  <key>StandardOutPath</key>
  <string>/tmp/com.coin.passcheck.batch.out.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/com.coin.passcheck.batch.err.log</string>
  <key>RunAtLoad</key>
  <false/>
</dict>
</plist>
PLIST

launchctl bootout "gui/$(id -u)" "$BACKUP_PLIST" >/dev/null 2>&1 || true
launchctl bootout "gui/$(id -u)" "$BATCH_PLIST" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$(id -u)" "$BACKUP_PLIST"
launchctl bootstrap "gui/$(id -u)" "$BATCH_PLIST"
launchctl enable "gui/$(id -u)/com.coin.passcheck.backup"
launchctl enable "gui/$(id -u)/com.coin.passcheck.batch"

echo "installed: $BACKUP_PLIST"
echo "installed: $BATCH_PLIST"
