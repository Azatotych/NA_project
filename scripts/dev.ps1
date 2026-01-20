param(
  [switch]$Wait,
  [switch]$NoOpen
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$logsDir = Join-Path $root "logs"
if (!(Test-Path $logsDir)) {
  New-Item -ItemType Directory -Path $logsDir | Out-Null
}

$backendOut = Join-Path $logsDir "backend.out.log"
$backendErr = Join-Path $logsDir "backend.err.log"
$frontendOut = Join-Path $logsDir "frontend.out.log"
$frontendErr = Join-Path $logsDir "frontend.err.log"

$backend = Start-Process -FilePath "python" `
  -ArgumentList "-m uvicorn backend.main:app --host 0.0.0.0 --port 8000" `
  -WorkingDirectory $root `
  -RedirectStandardOutput $backendOut `
  -RedirectStandardError $backendErr `
  -PassThru

$frontendDir = Join-Path $root "frontend"
if (!(Test-Path (Join-Path $frontendDir "node_modules"))) {
  Set-Location $frontendDir
  npm install
}

$npmCmd = Get-Command "npm.cmd" -ErrorAction SilentlyContinue
if (!$npmCmd) { $npmCmd = Get-Command "npm" -ErrorAction SilentlyContinue }
if (!$npmCmd) { throw "npm not found in PATH." }

$frontend = Start-Process -FilePath $npmCmd.Source `
  -ArgumentList "run dev" `
  -WorkingDirectory $frontendDir `
  -RedirectStandardOutput $frontendOut `
  -RedirectStandardError $frontendErr `
  -PassThru

Write-Host "Backend PID: $($backend.Id) | logs: $backendOut, $backendErr"
Write-Host "Frontend PID: $($frontend.Id) | logs: $frontendOut, $frontendErr"
if ($Wait) {
  Write-Host "Press Ctrl+C to stop both processes."
  try {
    Wait-Process -Id @($backend.Id, $frontend.Id)
  } finally {
    Get-Process -Id @($backend.Id, $frontend.Id) -ErrorAction SilentlyContinue | Stop-Process -Force
  }
} else {
  Write-Host "Started in background. Use logs to monitor. Re-run with -Wait to block in this console."
}

if (-not $NoOpen) {
  Start-Sleep -Seconds 2
  Start-Process "http://localhost:5173"
}
