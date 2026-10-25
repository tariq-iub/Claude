#!/usr/bin/env bash
# Cross-platform equivalent of publish.ps1: produces a self-contained win-x64 build and CuAnnotator-Release.zip.
# The resulting CuAnnotator.exe runs on Windows 10/11 x64 only.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"; VERSION="${1:-1.0.0}"
STAGE="$ROOT/artifacts/CuAnnotator-Release"
rm -rf "$STAGE" && mkdir -p "$STAGE"
[[ "${SKIP_TESTS:-0}" == "1" ]] || dotnet test CuAnnotator.sln -c Release
dotnet publish src/CuAnnotator.Desktop/CuAnnotator.Desktop.csproj -c Release -r win-x64 --self-contained true \
  -p:PublishSingleFile=false -p:Version="$VERSION" -o "$STAGE/Application"
mkdir -p "$STAGE/Source" && git archive HEAD | tar -x -C "$STAGE/Source"
cp -r documentation "$STAGE/Documentation"
cp -r samples/SampleProject "$STAGE/SampleProject"
cp -r tests "$STAGE/Tests"
cp -r benchmarks "$STAGE/Benchmarks"
find "$STAGE/Tests" "$STAGE/Benchmarks" -type d \( -name bin -o -name obj \) -prune -exec rm -rf {} +
cp -r scripts "$STAGE/BuildScripts"
cp README.md "$STAGE/"
(cd "$ROOT/artifacts" && rm -f CuAnnotator-Release.zip && zip -qr CuAnnotator-Release.zip CuAnnotator-Release)
echo "Release package: $ROOT/artifacts/CuAnnotator-Release.zip"
