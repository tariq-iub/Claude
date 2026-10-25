#!/usr/bin/env bash
# Restore, build and test (Linux/macOS builds the libraries, tests and WPF project via EnableWindowsTargeting;
# the WPF application itself can only run on Windows).
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-Release}"
dotnet restore CuAnnotator.sln
dotnet build CuAnnotator.sln -c "$CONFIG" --no-restore
dotnet test CuAnnotator.sln -c "$CONFIG" --no-build --logger "trx;LogFileName=test-results.trx" --results-directory artifacts/test-results
echo "Build and tests completed."
