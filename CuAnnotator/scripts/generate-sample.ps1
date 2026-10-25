Set-Location (Join-Path $PSScriptRoot "..")
dotnet run -c Release --project samples/CuAnnotator.SampleGenerator -- samples/SampleProject
