#!/usr/bin/env bash
set -euo pipefail

REMOTE="root@10.44.6.60"
PORT="10247"
KEY="/home/gwj/.ssh/id_ed25519_codex"
REMOTE_ROOT="/root/seva_vggt/outputs/test30_mvgen_vggt/downloads"
LOCAL_ROOT="/mnt/d/Desktop/seva_vggt/test30_mvgen_vggt"
PART_ROOT="$LOCAL_ROOT/download_parts"
LOG="$LOCAL_ROOT/download_chunks.log"
STATUS="$LOCAL_ROOT/download_status.txt"

SSH=(ssh -n -T -p "$PORT" -i "$KEY" -o StrictHostKeyChecking=no -o Compression=no -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -c chacha20-poly1305@openssh.com)

mkdir -p "$PART_ROOT"
exec >>"$LOG" 2>&1
echo "$(date -Is) START"

"${SSH[@]}" "$REMOTE" "cat '$REMOTE_ROOT/chunk_sizes.txt'" > "$LOCAL_ROOT/chunk_sizes.txt"
"${SSH[@]}" "$REMOTE" "cat '$REMOTE_ROOT/archive_sha256.txt'" > "$LOCAL_ROOT/archive_sha256.txt"

download_one() {
  local expected="$1"
  local rel="$2"
  local dst="$PART_ROOT/$rel"
  local tmp="$dst.partial"
  local actual=0

  mkdir -p "$(dirname "$dst")"
  if [[ -f "$dst" ]]; then
    actual=$(stat -c %s "$dst")
    if [[ "$actual" == "$expected" ]]; then
      echo "$(date -Is) SKIP $rel $actual"
      return 0
    fi
  fi

  echo "$(date -Is) GET $rel expected=$expected"
  : > "$tmp"
  until "${SSH[@]}" "$REMOTE" "cat '$REMOTE_ROOT/$rel'" > "$tmp"; do
    echo "$(date -Is) RETRY $rel"
    sleep 5
  done
  actual=$(stat -c %s "$tmp")
  if [[ "$actual" != "$expected" ]]; then
    echo "$(date -Is) SIZE_MISMATCH $rel expected=$expected actual=$actual"
    return 1
  fi
  mv -f "$tmp" "$dst"
  echo "$(date -Is) DONE $rel $actual"
}

while read -r expected rel; do
  [[ "$rel" == "total" ]] && continue
  [[ "$rel" == chunks_*/* ]] || continue
  download_one "$expected" "$rel"
  completed=$(find "$PART_ROOT" -type f ! -name '*.partial' | wc -l)
  printf 'downloading completed_chunks=%s current=%s updated=%s\n' "$completed" "$rel" "$(date -Is)" > "$STATUS"
done < "$LOCAL_ROOT/chunk_sizes.txt"

cat "$PART_ROOT"/chunks_results/part.* > "$LOCAL_ROOT/test30_results_images.tar.zst"
cat "$PART_ROOT"/chunks_real8/part.* > "$LOCAL_ROOT/test30_pointclouds_real8.tar.zst"
cat "$PART_ROOT"/chunks_mvgen/part.* > "$LOCAL_ROOT/test30_pointclouds_mvgen_1real7gen.tar.zst"

(
  cd "$LOCAL_ROOT"
  sha256sum -c archive_sha256.txt
)
printf 'downloaded_and_verified updated=%s\n' "$(date -Is)" > "$STATUS"
echo "$(date -Is) COMPLETE"
