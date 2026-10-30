param (
    [Parameter(Position=0)]
    [string]$Target = "run"
)

$PythonExe = "python"
if (Test-Path "$PSScriptRoot\venv\Scripts\python.exe") {
    $PythonExe = "$PSScriptRoot\venv\Scripts\python.exe"
}

& $PythonExe "$PSScriptRoot\scripts\make_runner.py" $Target $args
