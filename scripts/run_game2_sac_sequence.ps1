param(
    [double]$StopTimeDays = 1.0,
    [int]$IdleTimeoutSeconds = 45,
    [int]$WarmupSteps = 1000,
    [int]$BatchSize = 64,
    [int]$SaveFreq = 250
)

$ErrorActionPreference = "Stop"

$ProgRl = Split-Path -Parent $PSScriptRoot
$ThesisDir = Split-Path -Parent $ProgRl
$Bsm2Dir = Join-Path $ThesisDir "BSM2_R2019b"
$MatlabDir = Join-Path $ProgRl "matlab"
$LogDir = Join-Path $ProgRl "logs"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Run-Game2SacMode {
    param(
        [string]$Mode
    )

    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $PyOut = Join-Path $LogDir "game2_${Mode}_${Stamp}_stdout.txt"
    $PyErr = Join-Path $LogDir "game2_${Mode}_${Stamp}_stderr.txt"
    $CsvOut = Join-Path $LogDir "game2_${Mode}_${Stamp}_training_log.csv"

    Write-Host "[GAME2-SEQUENCE] Starting $Mode for $StopTimeDays day(s)."

    $Args = @(
        "-u",
        "agents\ctrl_game2_sac_coordinator.py",
        "--mode", $Mode,
        "--idle-timeout", "$IdleTimeoutSeconds",
        "--warmup-steps", "$WarmupSteps",
        "--batch-size", "$BatchSize",
        "--save-freq", "$SaveFreq",
        "--checkpoint-prefix", "game2_multiagent_sac"
    )
    $Coord = Start-Process -FilePath "python" `
        -ArgumentList $Args `
        -WorkingDirectory $ProgRl `
        -RedirectStandardOutput $PyOut `
        -RedirectStandardError $PyErr `
        -PassThru `
        -WindowStyle Hidden

    Start-Sleep -Seconds 2

    $MatlabCmd = "cd('$Bsm2Dir'); init_bsm2; cd('$MatlabDir'); GAME2_STOP_TIME=$StopTimeDays; GAME2_ABORT_ON_TIMEOUT=true; run('RL_main_game2.m');"
    & matlab -batch $MatlabCmd
    $MatlabExit = $LASTEXITCODE

    try {
        Wait-Process -Id $Coord.Id -Timeout ($IdleTimeoutSeconds + 120)
    } catch {
        Stop-Process -Id $Coord.Id -Force
        Write-Host "[GAME2-SEQUENCE] Coordinator for $Mode did not exit cleanly; stopped it."
    }

    if ($MatlabExit -ne 0) {
        throw "[GAME2-SEQUENCE] MATLAB failed for $Mode with exit code $MatlabExit"
    }

    $CurrentCsv = Join-Path $LogDir "game2_sac_training_log.csv"
    if (Test-Path $CurrentCsv) {
        Copy-Item -LiteralPath $CurrentCsv -Destination $CsvOut -Force
    }

    Write-Host "[GAME2-SEQUENCE] Finished $Mode."
    Write-Host "[GAME2-SEQUENCE] CSV: $CsvOut"
    Write-Host "[GAME2-SEQUENCE] stdout: $PyOut"
    Write-Host "[GAME2-SEQUENCE] stderr: $PyErr"
}

Run-Game2SacMode -Mode "sac-local"
Run-Game2SacMode -Mode "sac-game2"

Write-Host "[GAME2-SEQUENCE] Done."
