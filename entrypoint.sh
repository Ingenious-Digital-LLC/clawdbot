#!/bin/sh
# PolyClaw entrypoint — sync repo assets to mounted OpenClaw volume
# The volume at /home/node/.openclaw persists across deploys.
# We sync skills, workspace files, config, and cron jobs from the image.

OPENCLAW_DIR="/home/node/.openclaw"

# --- Skills: copy poly-* skills (skip if already present) ---
if [ -d /app/skills ]; then
  mkdir -p "$OPENCLAW_DIR/skills"
  for skill in /app/skills/poly-*; do
    if [ -d "$skill" ]; then
      skillname=$(basename "$skill")
      # Always overwrite to pick up code changes
      cp -r "$skill" "$OPENCLAW_DIR/skills/$skillname" 2>/dev/null || true
    fi
  done
fi

# --- Workspace: sync agent instructions (always overwrite for latest) ---
if [ -d /app/workspace ]; then
  mkdir -p "$OPENCLAW_DIR/workspace"
  for file in AGENTS.md SOUL.md HEARTBEAT.md; do
    if [ -f "/app/workspace/$file" ]; then
      cp "/app/workspace/$file" "$OPENCLAW_DIR/workspace/$file" 2>/dev/null || true
    fi
  done
fi

# --- Config: always overwrite to pick up config changes ---
if [ -f /app/config/openclaw.json ]; then
  cp /app/config/openclaw.json "$OPENCLAW_DIR/config.json" 2>/dev/null || true
fi

# --- Cron: deploy cron jobs (overwrite to pick up schedule changes) ---
if [ -f /app/config/cron/jobs.json ]; then
  mkdir -p "$OPENCLAW_DIR/cron"
  cp /app/config/cron/jobs.json "$OPENCLAW_DIR/cron/jobs.json" 2>/dev/null || true
fi

# --- Data: ensure trading data directories exist ---
mkdir -p "$OPENCLAW_DIR/workspace/data/research" 2>/dev/null || true
mkdir -p "$OPENCLAW_DIR/workspace/data/simulator" 2>/dev/null || true

# Execute the gateway
exec node "$@"
