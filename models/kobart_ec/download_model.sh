#!/usr/bin/env bash
# models/kobart_ec/download_model.sh
set -euo pipefail

TAG="ec-kobart-v0.1"
ASSET="model.safetensors"
URL="https://github.com/Seoyoung0519/LipSee/releases/download/${TAG}/${ASSET}"
DST="models/kobart_ec/${ASSET}"
SHA256_EXPECTED="142056d820f4ac177e0aee662a54cb600d9503474b1e07dbcf09ca7de17225d0"

mkdir -p "$(dirname "$DST")"

if [ -f "$DST" ]; then
  SHA256_ACTUAL="$(sha256sum "$DST" | awk '{print tolower($1)}')"
  if [ "$SHA256_ACTUAL" = "$SHA256_EXPECTED" ]; then
    echo "[OK] model exists & hash verified: $DST"
    exit 0
  else
    echo "[WARN] Hash mismatch. Re-downloading..."
    rm -f "$DST"
  fi
fi

curl -L "$URL" -o "$DST"

SHA256_ACTUAL="$(sha256sum "$DST" | awk '{print tolower($1)}')"
if [ "$SHA256_ACTUAL" != "$SHA256_EXPECTED" ]; then
  echo "[ERR] SHA256 mismatch: $SHA256_ACTUAL"
  exit 1
fi

echo "[DONE] Downloaded and verified $ASSET -> $DST"
