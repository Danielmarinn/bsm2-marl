param(
    [Parameter(Mandatory=$true)][ValidateSet('qec','qint','do','qw')][string]$Agent,
    [int]$StopTimeDays = 609
)

# Single-agent SAC baseline launcher (headless MATLAB, no GUI).
# Starts the Python controller, then runs the BSM2 closed loop via `matlab -batch`,
# passing SIMPLE_AGENT and SIMPLE_STOP_TIME into RL_main_simple.m. The single-agent
# Python controllers run an endless poll loop, so the Python process is stopped
# explicitly once MATLAB returns.
#
#   .\run_single_agent.ps1 -Agent qec               # full 609-day run
#   .\run_single_agent.ps1 -Agent qec -StopTimeDays 3   # short smoke

$ErrorActionPreference = "Stop"

$ProgRl    = Split-Path -Parent $PSScriptRoot
$ThesisDir = Split-Path -Parent $ProgRl
$Bsm2Dir   = Join-Path $ThesisDir "BSM2_R2019b"
$MatlabDir = Join-Path $ProgRl "matlab"
$LogDir    = Join-Path $ProgRl "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$PyScript = "agents\ctrl_sac_$Agent.py"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$PyOut = Join-Path $LogDir "single_${Agent}_${Stamp}_stdout.txt"
$PyErr = Join-Path $LogDir "single_${Agent}_${Stamp}_stderr.txt"

Write-Host "[SINGLE-AGENT] agent=$Agent  stop=$StopTimeDays d  (headless)"

# Clear stale comms flags so the controller and MATLAB rendezvous cleanly.
$CommsDir = Join-Path $ProgRl "comms"
if (Test-Path $CommsDir) {
    $StaleComms = Get-ChildItem -LiteralPath $CommsDir -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match 'flag_|state.csv|action.csv' }
    if ($StaleComms) {
        $TrashDir = Join-Path $ThesisDir "_trash_pending\single_agent_comms_$Stamp"
        New-Item -ItemType Directory -Force -Path $TrashDir | Out-Null
        $StaleComms | ForEach-Object {
            Move-Item -LiteralPath $_.FullName -Destination (Join-Path $TrashDir $_.Name) -Force
        }
        Write-Host "[SINGLE-AGENT] Moved stale comms to $TrashDir"
    }
}

$OldPythonIoEncoding = $env:PYTHONIOENCODING
$env:PYTHONIOENCODING = "utf-8"
$Py = Start-Process -FilePath "python" `
    -ArgumentList @("-u", $PyScript) `
    -WorkingDirectory $ProgRl `
    -RedirectStandardOutput $PyOut `
    -RedirectStandardError $PyErr `
    -PassThru -WindowStyle Hidden
$env:PYTHONIOENCODING = $OldPythonIoEncoding
Start-Sleep -Seconds 2

$MatlabCmd = "cd('$Bsm2Dir'); init_bsm2; cd('$MatlabDir'); SIMPLE_AGENT='$Agent'; SIMPLE_STOP_TIME=$StopTimeDays; run('RL_main_simple.m');"
& matlab -batch $MatlabCmd
$MatlabExit = $LASTEXITCODE

Stop-Process -Id $Py.Id -Force -ErrorAction SilentlyContinue

Write-Host "[SINGLE-AGENT] MATLAB exit=$MatlabExit"
Write-Host "[SINGLE-AGENT] Python stdout: $PyOut"
Write-Host "[SINGLE-AGENT] Python stderr: $PyErr"
if ($MatlabExit -ne 0) { throw "[SINGLE-AGENT] MATLAB exited non-zero ($MatlabExit). Check the stderr file." }
Write-Host "[SINGLE-AGENT] Done. Training log written under $LogDir."
