<#
.SYNOPSIS
    Confidence-Aware Digital Twin EMS — Live Hardware & System Launcher
.DESCRIPTION
    Starts the full live stack:
      1. MQTT Broker (0.0.0.0:1883)
      2. Hardware Ingestion Pipeline (config.hardware.yaml) connected to ESP32 node_bench_agg
      3. FastAPI Backend Server (port 8000)
      4. React 19 Frontend Dashboard (port 5173)
.EXAMPLE
    .\run.ps1
.EXAMPLE
    .\run.ps1 -NoFrontend
#>

param (
    [switch]$NoFrontend,
    [switch]$Demo
)

$ErrorActionPreference = "Stop"

# Use virtual environment python if present
$PythonExe = "python"
if (Test-Path "venv\Scripts\python.exe") {
    $PythonExe = "venv\Scripts\python.exe"
} elseif (Test-Path ".venv\Scripts\python.exe") {
    $PythonExe = ".venv\Scripts\python.exe"
}

if ($Demo) {
    Write-Host "Starting Digital Twin EMS in Software Demo Mode..." -ForegroundColor Cyan
    & $PythonExe scripts/demo_full_system.py --demo
} else {
    $ArgsList = @("scripts/run_hardware_system.py")
    if ($NoFrontend) {
        $ArgsList += "--no-frontend"
    }
    & $PythonExe $ArgsList
}
