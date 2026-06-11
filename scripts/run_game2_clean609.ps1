param(
    [int]$StopTimeDays      = 609,
    [int]$Seed              = 7,
    [string]$CheckpointPrefix = "game2_clean609v2",
    [int]$IdleTimeoutSeconds = 300
)

# Clean 609-day TRAINING run of the recurrent CTDE-SAC + G2ANet controller.
# Launched HEADLESS via `matlab -batch` (no GUI to lag/crash) so that, together
# with the auto-save block in RL_main_game2.m, the full plant trajectory is
# always written to logs\game2_traj_<stamp>.mat at clean completion.
#
# Canonical hyperparameters of RecurrentCTDESACConfig are passed explicitly
# because the coordinator's argparse defaults (e.g. warmup 1000) would otherwise
# override the config (warmup 5000).

$ErrorActionPreference = "Stop"

$ProgRl    = Split-Path -Parent $PSScriptRoot
$ThesisDir = Split-Path -Parent $ProgRl
$Bsm2Dir   = Join-Path $ThesisDir "BSM2_R2019b"
$MatlabDir = Join-Path $ProgRl "matlab"
$LogDir    = Join-Path $ProgRl "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Mode  = "sac-ctde-rnn"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$PyOut = Join-Path $LogDir "game2_${Mode}_${Stamp}_stdout.txt"
$PyErr = Join-Path $LogDir "game2_${Mode}_${Stamp}_stderr.txt"

Write-Host "[CLEAN609] mode=$Mode  stop=$StopTimeDays d  seed=$Seed  prefix=$CheckpointPrefix"

$Args = @(
    "-u",
    "agents\ctrl_game2_sac_coordinator.py",
    "--mode", $Mode,
    "--warmup-steps", "5000",
    "--batch-size", "32",
    "--seq-len", "32",
    "--burn-in", "8",
    "--save-freq", "1000",
    "--idle-timeout", "$IdleTimeoutSeconds",
    "--seed", "$Seed",
    "--checkpoint-prefix", $CheckpointPrefix
)
$Coord = Start-Process -FilePath "python" `
    -ArgumentList $Args `
    -WorkingDirectory $ProgRl `
    -RedirectStandardOutput $PyOut `
    -RedirectStandardError $PyErr `
    -PassThru `
    -WindowStyle Hidden

Start-Sleep -Seconds 3

# Headless MATLAB: blocks here for the full run (~3 days), then exits cleanly.
$MatlabCmd = "cd('$Bsm2Dir'); init_bsm2; cd('$MatlabDir'); GAME2_STOP_TIME=$StopTimeDays; GAME2_ABORT_ON_TIMEOUT=true; run('RL_main_game2.m');"
& matlab -batch $MatlabCmd
$MatlabExit = $LASTEXITCODE

# After MATLAB finishes, let the coordinator idle-out and flush its final checkpoint.
try {
    Wait-Process -Id $Coord.Id -Timeout ($IdleTimeoutSeconds + 180)
} catch {
    Stop-Process -Id $Coord.Id -Force
    Write-Host "[CLEAN609] Coordinator did not exit on its own; stopped it."
}

Write-Host "[CLEAN609] MATLAB exit code: $MatlabExit"
Write-Host "[CLEAN609] Trajectory .mat auto-saved under $LogDir as game2_traj_*.mat (if completed cleanly)."
Write-Host "[CLEAN609] Python stdout: $PyOut"
Write-Host "[CLEAN609] Python stderr: $PyErr"
if ($MatlabExit -ne 0) {
    throw "[CLEAN609] MATLAB exited non-zero ($MatlabExit) - run likely aborted; check stderr."
}
