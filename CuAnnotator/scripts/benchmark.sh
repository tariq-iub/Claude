#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
dotnet run -c Release --project benchmarks/CuAnnotator.Benchmarks -- benchmarks/results "$@"
