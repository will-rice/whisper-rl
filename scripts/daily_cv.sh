#!/usr/bin/env bash
# Daily Common Voice refresh: download newly available locales (within the
# 30/day API limit), extract any new archives, and rebuild the streamable
# index. Idempotent and resumable — safe to run every day. Intended for cron:
#
#   0 1 * * * /bin/bash $HOME/projects/whisper-rl/scripts/daily_cv.sh \
#       >> /data/common_voice_26/daily.log 2>&1
#
# Downloading and ingesting need no GPU/FFmpeg; training is a separate step.
set -uo pipefail

REPO="$HOME/projects/whisper-rl"
DATA="/data/common_voice_26"
UV="$HOME/.local/bin/uv"

cd "$REPO"
echo "=== daily_cv start $(date -u +%FT%TZ) ==="

# 1. Pull up to the daily limit of accepted, not-yet-downloaded locales.
"$UV" run download-cv "$DATA" --max 30 || true

# 2. Extract any archive we have not extracted yet (marker per archive).
mkdir -p "$DATA/extracted"
for tar in "$DATA"/archives/*.tar.gz; do
  [ -e "$tar" ] || continue
  if [ ! -f "$tar.extracted" ]; then
    echo "extracting $(basename "$tar")"
    tar xzf "$tar" -C "$DATA/extracted" && touch "$tar.extracted"
  fi
done

# 3. Rebuild the index over every extracted locale (idempotent).
root=$(find "$DATA/extracted" -maxdepth 1 -type d -name 'cv-corpus-*' | sort | tail -1)
if [ -n "$root" ]; then
  "$UV" run ingest-cv "$root" "$DATA/index"
fi

echo "=== daily_cv done $(date -u +%FT%TZ) ==="
