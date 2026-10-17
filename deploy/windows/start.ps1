# Kassa Sentinel - production start on the Windows laptop (owner: A17). ASCII-only on purpose.
#   powershell -ExecutionPolicy Bypass -File deploy\windows\start.ps1            # quick tunnel
#   powershell -ExecutionPolicy Bypass -File deploy\windows\start.ps1 -Named     # named tunnel (TUNNEL_TOKEN in deploy\.env)
#   ... -NoTunnel                                                                  # local only
# Order: env from deploy\.env -> offline + process-local netguard -> weights verify -> keep system awake
# (display may turn off) -> uvicorn (ONE worker, ONE GPU job at a time) under a restart loop ->
# READY only after a real trial inference (healthcheck ready) -> only then cloudflared.
# No global firewall / power-plan changes. Ctrl+C stops everything.
param(
    [int]$Port = 8000,
    [string]$BindHost = "127.0.0.1",
    [string]$App = "backend.api.main:app",
    [switch]$Named,
    [switch]$NoTunnel,
    [int]$ReadyTimeoutSec = 600
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Repo
$Py = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { Write-Host "Run deploy\windows\setup.ps1 first" -ForegroundColor Red; exit 1 }
$Logs = Join-Path $Repo "logs"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
function Log($msg) { $line = "{0} {1}" -f (Get-Date -Format s), $msg; Write-Host $line; Add-Content (Join-Path $Logs "start.log") $line }

# --- env: deploy\.env (container paths /app/... are mapped to the repo) -----------------------------
$EnvFile = Join-Path $Repo "deploy\.env"
if (Test-Path $EnvFile) {
    foreach ($raw in Get-Content $EnvFile) {
        $line = $raw.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) { continue }
        $k, $v = $line.Split("=", 2)
        $v = $v.Trim().Trim('"').Trim("'")
        if ($v.StartsWith("/app/")) { $v = Join-Path $Repo ($v.Substring(5) -replace "/", "\") }
        [Environment]::SetEnvironmentVariable($k.Trim(), $v, "Process")
    }
} else { Log "WARN deploy\.env missing - copy deploy\.env.example" }
foreach ($kv in @{
        "HF_HUB_OFFLINE" = "1"; "TRANSFORMERS_OFFLINE" = "1"; "HF_DATASETS_OFFLINE" = "1"; "HF_HUB_DISABLE_TELEMETRY" = "1";
        "DO_NOT_TRACK" = "1"; "SENTINEL_WORKERS" = "1"; "SENTINEL_PORT" = "$Port"; "PYTHONUTF8" = "1";
        "SENTINEL_WEIGHTS" = (Join-Path $Repo "weights"); "SENTINEL_DATA_DIR" = (Join-Path $Repo "data");
        "SENTINEL_CACHE_DIR" = (Join-Path $Repo "cache"); "SENTINEL_LOG_DIR" = $Logs;
        "SENTINEL_FRONTEND_DIST" = (Join-Path $Repo "frontend\dist") }.GetEnumerator()) {
    if (-not [Environment]::GetEnvironmentVariable($kv.Key, "Process")) { [Environment]::SetEnvironmentVariable($kv.Key, $kv.Value, "Process") }
}
$env:HF_HUB_OFFLINE = "1"; $env:SENTINEL_WORKERS = "1"   # never overridden: runtime downloads nothing, one GPU worker
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $env:SENTINEL_WEIGHTS "hf" }
if (-not $env:HF_HUB_CACHE) { $env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub" }
if (-not $env:SENTINEL_NETGUARD) { $env:SENTINEL_NETGUARD = "block" }
$env:SENTINEL_NETGUARD_LOG = Join-Path $Logs "netguard.jsonl"
$env:PYTHONPATH = "$Repo;$Repo\deploy\netguard"
if (-not $env:SENTINEL_API_KEYS) { Log "ERROR SENTINEL_API_KEYS empty (deploy\.env)"; exit 1 }

# --- gates ---------------------------------------------------------------------------------------
if ($env:SENTINEL_MOCK -eq "1") { Log "WARN SENTINEL_MOCK=1 - fake models, NOT for the jury" }
else {
    & $Py deploy\weights.py verify --fast
    if ($LASTEXITCODE -ne 0) { Log "ERROR weights verify failed - run setup.ps1 (weights prepare)"; exit 1 }
}

# --- keep the SYSTEM awake while this script runs (display is allowed to turn off) ------------------
Add-Type -Namespace Sentinel -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$ES_CONTINUOUS = [uint32]"0x80000000"; $ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
[void][Sentinel.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
Log "system sleep blocked for this session (lid close still follows the power plan - keep the lid open)"

$Url = "http://127.0.0.1:$Port"
$script:Api = $null; $script:Tunnel = $null; $script:Ready = $false; $script:Restarts = 0

function Start-Api {
    $uvArgs = @("-m", "uvicorn", $App, "--host", $BindHost, "--port", "$Port", "--workers", "1", "--timeout-keep-alive", "75")
    $script:Api = Start-Process -FilePath $Py -ArgumentList $uvArgs -NoNewWindow -PassThru `
        -RedirectStandardOutput (Join-Path $Logs "api.out.log") -RedirectStandardError (Join-Path $Logs "api.err.log")
    Log "api started pid=$($script:Api.Id) $App on ${BindHost}:$Port"
}
function Start-Tunnel {
    if ($NoTunnel) { return }
    $cf = (Get-Command cloudflared -ErrorAction SilentlyContinue)
    if (-not $cf) { Log "WARN cloudflared not installed - no public URL"; return }
    if ($Named) {
        if (-not $env:TUNNEL_TOKEN) { Log "ERROR -Named needs TUNNEL_TOKEN in deploy\.env"; return }
        $targs = @("tunnel", "--no-autoupdate", "run", "--token", $env:TUNNEL_TOKEN)
    } else {
        $targs = @("tunnel", "--no-autoupdate", "--url", $Url)
    }
    if ($env:CLOUDFLARED_PROTOCOL) { $targs = @("tunnel", "--protocol", $env:CLOUDFLARED_PROTOCOL) + $targs[1..($targs.Length - 1)] }
    $script:Tunnel = Start-Process -FilePath $cf.Source -ArgumentList $targs -NoNewWindow -PassThru `
        -RedirectStandardOutput (Join-Path $Logs "cloudflared.out.log") -RedirectStandardError (Join-Path $Logs "cloudflared.err.log")
    Log "cloudflared started pid=$($script:Tunnel.Id)"
}
function Get-QuickUrl {
    $f = Join-Path $Logs "cloudflared.err.log"
    if (Test-Path $f) {
        $m = Select-String -Path $f -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" | Select-Object -Last 1
        if ($m) { return $m.Matches[0].Value }
    }
    return $null
}

try {
    Start-Api
    $deadline = (Get-Date).AddSeconds($ReadyTimeoutSec)
    $liveFails = 0; $announced = $false
    while ($true) {
        Start-Sleep -Seconds 5
        if ($script:Api.HasExited) {
            $script:Restarts++
            $delay = [Math]::Min(60, 5 * $script:Restarts)
            Log "api exited code=$($script:Api.ExitCode) - restart #$($script:Restarts) in ${delay}s (see logs\api.err.log)"
            Start-Sleep -Seconds $delay; Start-Api; $script:Ready = $false; continue
        }
        & $Py deploy\healthcheck.py live --url $Url --quiet
        if ($LASTEXITCODE -ne 0) {
            $liveFails++
            if ($script:Ready -and $liveFails -ge 6) { Log "api not answering 30s - restarting"; Stop-Process -Id $script:Api.Id -Force; $liveFails = 0 }
            continue
        }
        $liveFails = 0
        if (-not $script:Ready) {
            Log "api is live - running REAL trial inference (healthcheck ready)"
            & $Py deploy\healthcheck.py ready --url $Url --limit 300 --json | Tee-Object -FilePath (Join-Path $Logs "ready.json") | Out-Null
            if ($LASTEXITCODE -eq 0) {
                $script:Ready = $true; Log "READY (trial inference passed, see logs\ready.json)"
                if (-not $script:Tunnel -or $script:Tunnel.HasExited) { Start-Tunnel }
            } elseif ((Get-Date) -gt $deadline) {
                Log "ERROR not ready after ${ReadyTimeoutSec}s - tunnel NOT opened (logs\ready.json, logs\api.err.log)"; $deadline = (Get-Date).AddSeconds($ReadyTimeoutSec)
            }
            continue
        }
        if ($script:Tunnel -and $script:Tunnel.HasExited) { Log "cloudflared exited - restarting"; Start-Sleep 3; Start-Tunnel; $announced = $false }
        if (-not $announced -and $script:Tunnel -and -not $Named) {
            $u = Get-QuickUrl
            if ($u) { Log "PUBLIC URL: $u  (check: python deploy\tunnel_probe.py real --url $u)"; Set-Content (Join-Path $Logs "public_url.txt") $u; $announced = $true }
        }
    }
} finally {
    foreach ($p in @($script:Tunnel, $script:Api)) { if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } }
    [void][Sentinel.Power]::SetThreadExecutionState($ES_CONTINUOUS)
    Log "stopped"
}
