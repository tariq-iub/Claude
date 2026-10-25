# Self-contained Windows x64 release and CuAnnotator-Release.zip.
# Run on Windows (or any OS with the .NET 10 SDK; WPF publishing works cross-platform with EnableWindowsTargeting).
param([string]$Version = "1.0.0", [switch]$SkipTests)
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root
$stage = Join-Path $root "artifacts/CuAnnotator-Release"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $stage | Out-Null

if (-not $SkipTests) { dotnet test CuAnnotator.sln -c Release }

dotnet publish src/CuAnnotator.Desktop/CuAnnotator.Desktop.csproj -c Release -r win-x64 --self-contained true `
    -p:PublishSingleFile=false -p:PublishReadyToRun=true -p:Version=$Version -o (Join-Path $stage "Application")

# Source (tracked files only), documentation, samples, tests, benchmarks, scripts.
git archive --format=zip -o (Join-Path $root "artifacts/source.zip") HEAD
Expand-Archive (Join-Path $root "artifacts/source.zip") -DestinationPath (Join-Path $stage "Source") -Force
Remove-Item (Join-Path $root "artifacts/source.zip")
Copy-Item -Recurse documentation (Join-Path $stage "Documentation")
Copy-Item -Recurse samples/SampleProject (Join-Path $stage "SampleProject")
Copy-Item -Recurse tests (Join-Path $stage "Tests") -Exclude bin,obj
Copy-Item -Recurse benchmarks (Join-Path $stage "Benchmarks") -Exclude bin,obj
Copy-Item -Recurse scripts (Join-Path $stage "BuildScripts")
Copy-Item README.md $stage

$zip = Join-Path $root "artifacts/CuAnnotator-Release.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zip
Write-Host "Release package: $zip" -ForegroundColor Green
Write-Host "Run: $stage/Application/CuAnnotator.exe"
