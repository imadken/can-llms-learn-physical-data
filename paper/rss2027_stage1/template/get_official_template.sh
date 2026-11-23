#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARCHIVE="$ROOT/template/paper-template-latex.tar.gz"
URL="https://roboticsconference.org/docs/paper-template-latex.tar.gz"
EXPECTED="ffc8b8053bd218ba9be4c93a569de4c2f36c7f70ae8d2245c747611bed697271"
mkdir -p "$ROOT/template"
if [ ! -s "$ARCHIVE" ]; then
  echo 'Downloading official RSS LaTeX template...'
  curl --fail --location --retry 3 --silent --show-error "$URL" -o "$ARCHIVE"
fi
ACTUAL="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "ERROR: RSS template archive checksum changed: $ACTUAL" >&2
  echo 'Inspect the current conference website and update the pinned checksum after verification.' >&2
  exit 1
fi
tar -xzf "$ARCHIVE" -C "$ROOT/template" IEEEtran.cls
if ! grep -q '2007/03/05 version V1.7a' "$ROOT/template/IEEEtran.cls"; then
  echo 'ERROR: unexpected IEEEtran class version.' >&2; exit 1
fi
echo "Verified RSS IEEEtran.cls: $ROOT/template/IEEEtran.cls"
