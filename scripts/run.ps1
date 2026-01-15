$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$runsDir = Join-Path $root "runs"
if (-not (Test-Path $runsDir)) {
    New-Item -ItemType Directory -Path $runsDir | Out-Null
}

$venvActivate = Join-Path $root "venv\\Scripts\\Activate.ps1"
if (Test-Path $venvActivate) {
    & $venvActivate
}

$backendLog = Join-Path $runsDir "backend.log"
$frontendLog = Join-Path $runsDir "frontend.log"

$backendCmd = "cd `"$root`"; uvicorn backend.main:app --host 0.0.0.0 --port 8000 *> `"$backendLog`""
$frontendDir = Join-Path $root "frontend"
$frontendCmd = "cd `"$frontendDir`"; if (-not (Test-Path node_modules)) { npm install }; npm run dev -- --host 0.0.0.0 --port 5173 *> `"$frontendLog`""

Start-Process -FilePath "powershell" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -Command `"$backendCmd`"" | Out-Null
Start-Process -FilePath "powershell" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -Command `"$frontendCmd`"" | Out-Null

Start-Sleep -Seconds 2
Start-Process "http://localhost:5173/"
