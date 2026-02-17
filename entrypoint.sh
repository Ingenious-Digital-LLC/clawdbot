#!/bin/sh
# Copy poly-* skills from image to mounted volume if not already there
if [ -d /app/skills ]; then
  for skill in /app/skills/poly-*; do
    if [ -d "$skill" ]; then
      skillname=$(basename "$skill")
      if [ ! -d "/home/node/.openclaw/skills/$skillname" ]; then
        cp -r "$skill" "/home/node/.openclaw/skills/$skillname" 2>/dev/null || true
      fi
    fi
  done
fi

# Execute the original command
exec node "$@"
