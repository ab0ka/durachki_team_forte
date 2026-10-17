# Kassa Sentinel - locality proof on Windows (owner: A17). ASCII-only.
# Runs the REAL worker (backend.pipeline.run) on local weights with HF offline and the process-local
# netguard in block mode. No firewall changes. Separate from the public HTTP check (tunnel_probe.py).
#   powershell -ExecutionPolicy Bypass -File deploy\windows\offline_smoke.ps1
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Repo
$Py = Join-Path $Repo ".venv\Scripts\python.exe"
$env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"; $env:SENTINEL_MOCK = "0"
$env:SENTINEL_NETGUARD = "block"; $env:PYTHONPATH = "$Repo;$Repo\deploy\netguard"
if (-not $env:SENTINEL_WEIGHTS) { $env:SENTINEL_WEIGHTS = Join-Path $Repo "weights" }
& $Py deploy\offline_worker_smoke.py @args
exit $LASTEXITCODE
