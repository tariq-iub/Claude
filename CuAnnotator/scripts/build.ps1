# Restore, build and test the whole solution (Windows PowerShell 5+ or PowerShell 7).
param([string]$Configuration = "Release")
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
dotnet restore CuAnnotator.sln
dotnet build CuAnnotator.sln -c $Configuration --no-restore
dotnet test CuAnnotator.sln -c $Configuration --no-build --logger "trx;LogFileName=test-results.trx" --results-directory artifacts/test-results
Write-Host "Build and tests completed. Test results: artifacts/test-results" -ForegroundColor Green
