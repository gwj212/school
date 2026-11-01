#!/usr/bin/env bash
set -euo pipefail

REMOTE="root@10.44.6.60"
PORT="10247"
KEY="/home/gwj/.ssh/id_ed25519_codex"
REMOTE_ROOT="/root/seva_vggt/outputs/test30_mvgen_vggt/downloads"
LOCAL_ROOT="/mnt/d/Desktop/seva_vggt/test30_mvgen_vggt"
LOG="$LOCAL_ROOT/download_archives.log"
STATUS="$LOCAL_ROOT/download_status.txt"
SSH=(ssh -n -T -p "$PORT" -i "$KEY" -o StrictHostKeyChecking=no -o Compression=no -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -c chacha20-poly1305@openssh.com)

mkdir -p "$LOCAL_ROOT"
exec >>"$LOG" 2>&1
echo "$(date -Is) START_RESUMABLE_ARCHIVES"

download_archive() {
  local file="$1"
  local expected="$2"
  local dst="$LOCAL_ROOT/$file"
  local tmp="$dst.partial"
  local current=0
  local offset=1

  if [[ -f "$dst" ]] && [[ "$(stat -c %s "$dst")" == "$expected" ]]; then
    echo "$(date -Is) SKIP_COMPLETE $file $expected"
    return 0
  fi

  if [[ "$file" == "test30_results_images.tar.zst" ]] && [[ ! -f "$tmp" ]] && [[ -f "$LOCAL_ROOT/download_parts/chunks_results/part.0000" ]]; then
    cp "$LOCAL_ROOT/download_parts/chunks_results/part.0000" "$tmp"
    echo "$(date -Is) SEEDED $file $(stat -c %s "$tmp")"
  fi
  [[ -f "$tmp" ]] || : > "$tmp"

  while true; do
    current=$(stat -c %s "$tmp")
    if (( current == expected )); then
      mv -f "$tmp" "$dst"
      echo "$(date -Is) DONE_ARCHIVE $file $current"
      return 0
    fi
    if (( current > expected )); then
      echo "$(date -Is) OVERSIZE $file expected=$expected actual=$current"
      return 1
    fi
    offset=$((current + 1))
    printf 'downloading archive=%s bytes=%s expected=%s updated=%s\n' "$file" "$current" "$expected" "$(date -Is)" > "$STATUS"
    echo "$(date -Is) STREAM $file offset=$offset expected=$expected"
    if ! "${SSH[@]}" "$REMOTE" "tail -c +$offset '$REMOTE_ROOT/$file'" >> "$tmp"; then
      echo "$(date -Is) RETRY_STREAM $file current=$(stat -c %s "$tmp")"
      sleep 5
    fi
  done
}

sizes="$LOCAL_ROOT/chunk_sizes.txt"
[[ -f "$sizes" ]]
results_size=$(awk '$2 ~ /^chunks_results\// {s+=$1} END {printf "%.0f",s}' "$sizes")
real8_size=$(awk '$2 ~ /^chunks_real8\// {s+=$1} END {printf "%.0f",s}' "$sizes")
mvgen_size=$(awk '$2 ~ /^chunks_mvgen\// {s+=$1} END {printf "%.0f",s}' "$sizes")

download_archive test30_results_images.tar.zst "$results_size"
download_archive test30_pointclouds_real8.tar.zst "$real8_size"
download_archive test30_pointclouds_mvgen_1real7gen.tar.zst "$mvgen_size"

(
  cd "$LOCAL_ROOT"
  sha256sum -c archive_sha256.txt
)
printf 'downloaded_and_verified updated=%s\n' "$(date -Is)" > "$STATUS"
echo "$(date -Is) COMPLETE_RESUMABLE_ARCHIVES"
