param([switch]$Quick)
Set-Location (Join-Path $PSScriptRoot "..")
$extra = @(); if ($Quick) { $extra += "--quick" }
dotnet run -c Release --project benchmarks/CuAnnotator.Benchmarks -- benchmarks/results @extra
