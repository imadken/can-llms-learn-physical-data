#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLASS="$ROOT/template/IEEEtran.cls"
ARCHIVE="$ROOT/template/paper-template-latex.tar.gz"
URL="https://roboticsconference.org/docs/paper-template-latex.tar.gz"
EXPECTED_ARCHIVE="ffc8b8053bd218ba9be4c93a569de4c2f36c7f70ae8d2245c747611bed697271"
EXPECTED_CLASS="d63d4bf27f29603a733d06e27ba5fbb40488acf9d1d696354a88b886758f70a7"
if [ ! -f "$CLASS" ]; then
  if [ ! -s "$ARCHIVE" ]; then
    curl --fail --location --retry 3 --silent --show-error "$URL" -o "$ARCHIVE"
  fi
  ACTUAL_ARCHIVE="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
  if [ "$ACTUAL_ARCHIVE" != "$EXPECTED_ARCHIVE" ]; then
    echo "RSS template archive checksum mismatch: $ACTUAL_ARCHIVE" >&2
    exit 1
  fi
  tar -xzf "$ARCHIVE" -C "$ROOT/template" IEEEtran.cls
fi
ACTUAL_CLASS="$(sha256sum "$CLASS" | awk '{print $1}')"
if [ "$ACTUAL_CLASS" != "$EXPECTED_CLASS" ]; then
  echo "RSS IEEEtran.cls checksum mismatch: $ACTUAL_CLASS" >&2
  exit 1
fi
grep -q '2007/03/05 version V1.7a' "$CLASS"
echo "Verified official RSS IEEEtran.cls V1.7a"
