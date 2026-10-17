# Kassa Sentinel - one-time setup on the Windows 11 laptop (owner: A17). ASCII-only on purpose
# (Windows PowerShell 5.1 reads BOM-less files as ANSI).
#   powershell -ExecutionPolicy Bypass -File deploy\windows\setup.ps1            # GPU (cu124)
#   powershell -ExecutionPolicy Bypass -File deploy\windows\setup.ps1 -Cpu       # CPU fallback
# Steps: uv -> Python 3.11 venv -> merge requirements (conflicts stop here) -> lock via real resolver
# (uv pip compile, target windows-cu124) -> exact install from lock (hashes) -> GPU probe -> .env ->
# weights prepare (SEPARATE online step) + verify -> optional frontend build.
# Does NOT change firewall or global power settings.
param(
    [switch]$Cpu,
    [switch]$Relock,          # rebuild deploy\locks\<target>.lock.txt even if it exists
    [switch]$SkipWeights,
    [switch]$SkipFrontend,
    [string]$PythonVersion = "3.11"
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Repo
function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "XX  $msg" -ForegroundColor Red; exit 1 }

Step "uv"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    powershell -ExecutionPolicy ByPass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Fail "uv not found after install" }
}
uv --version

Step "Python $PythonVersion venv (.venv)"
uv python install $PythonVersion
if (-not (Test-Path ".venv\Scripts\python.exe")) { uv venv .venv --python $PythonVersion }
$Py = Join-Path $Repo ".venv\Scripts\python.exe"
uv pip install --python $Py "packaging>=23"   # bootstrap for merge_requirements; sync reconciles later

$Target = if ($Cpu) { "windows-cpu" } else { "windows-cu124" }
$Backend = if ($Cpu) { "cpu" } else { "cu124" }
$ReqProfile = if ($Cpu) { "cpu" } else { "gpu" }
$Lock = "deploy\locks\$Target.lock.txt"

Step "merge requirements of all modules (no version guessing)"
& $Py deploy\merge_requirements.py --profile $ReqProfile
if ($LASTEXITCODE -ne 0) { Fail "requirements conflicts/banned packages (see report). Resolve with module owners / A01." }

if ($Relock -or -not (Test-Path $Lock)) {
    Step "lock via resolver: uv pip compile -> $Lock"
    & $Py deploy\merge_requirements.py --profile $ReqProfile --lock $Target
    if ($LASTEXITCODE -ne 0) { Fail "resolver failed - this is a real conflict, do not pin by hand" }
    Write-Host "Commit $Lock so every machine installs the same versions."
}

Step "exact install from lock (hashes required)"
uv pip sync $Lock --python $Py --torch-backend $Backend --require-hashes
if ($LASTEXITCODE -ne 0) { Fail "uv pip sync failed" }

Step "GPU probe on THIS machine"
$probeArgs = @("deploy\gpu_probe.py")
if ($Cpu) { $probeArgs += "--allow-cpu" }
& $Py @probeArgs
if ($LASTEXITCODE -ne 0) { Fail "GPU probe failed (logs\gpu_probe.json). Check NVIDIA driver (nvidia-smi) and the torch build." }

Step "deploy\.env (secrets stay local, never in git)"
if (-not (Test-Path "deploy\.env")) {
    Copy-Item "deploy\.env.example" "deploy\.env"
    $bytes = New-Object byte[] 24
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $key = ([Convert]::ToBase64String($bytes)) -replace '[+/=]', ''
    (Get-Content "deploy\.env") -replace '^SENTINEL_API_KEYS=.*$', "SENTINEL_API_KEYS=$key" | Set-Content "deploy\.env" -Encoding ASCII
    Write-Host "Generated API key -> deploy\.env (SENTINEL_API_KEYS). Give it to the jury out of band."
}

if (-not $SkipWeights) {
    Step "weights: prepare (online, separate step) + verify (offline)"
    $env:HF_HUB_OFFLINE = "0"
    & $Py deploy\weights.py prepare
    if ($LASTEXITCODE -ne 0) { Fail "weights prepare failed (license red flag / disk / network) - see above" }
    $env:HF_HUB_OFFLINE = "1"
    & $Py deploy\weights.py verify
    if ($LASTEXITCODE -ne 0) { Fail "weights verify failed" }
}

if (-not $SkipFrontend -and (Test-Path "frontend\package.json")) {
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        Step "frontend build (A15)"
        Push-Location frontend
        if (Test-Path "package-lock.json") { npm ci } else { npm install }
        npm run build
        Pop-Location
    } else { Write-Host "npm not found - UI not built (winget install OpenJS.NodeJS.LTS)" -ForegroundColor Yellow }
}

if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    Write-Host "cloudflared not found: winget install --id Cloudflare.cloudflared" -ForegroundColor Yellow
}
Step "done. Next: deploy\windows\offline_smoke.ps1, then deploy\windows\start.ps1"
