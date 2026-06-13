# Watchdog for Task 2.5 long-running sweep.
# Polls every 480s (8 min). Emits ONE heartbeat line per check.
# Auto-restarts the parent run_task25.py if dead (cached pkls skipped on resume).

$restarts    = 0
$maxRestarts = 5
$totalRuns   = 30

while ($true) {
    Start-Sleep -Seconds 480

    $py = Get-CimInstance Win32_Process |
        Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*run_task25*' } |
        Select-Object -First 1

    $pklCount = @(Get-ChildItem 'C:\Users\Admin\brain-organoid-rl\ablation_results\task25\T25_*.pkl' -ErrorAction SilentlyContinue).Count

    $lastFile = Get-ChildItem 'C:\Users\Admin\brain-organoid-rl\ablation_results\task25\' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $minAgo = if ($lastFile) { [Math]::Round(((Get-Date) - $lastFile.LastWriteTime).TotalMinutes, 1) } else { -1 }

    $ts = Get-Date -Format 'HH:mm'

    if ($py) {
        Write-Host "[HB $ts] ALIVE pid=$($py.ProcessId) pkls=$pklCount/$totalRuns last_write=${minAgo}min_ago"
    } elseif ($pklCount -ge $totalRuns) {
        Write-Host "[HB $ts] DONE pkls=$pklCount/$totalRuns watchdog exiting"
        break
    } else {
        if ($restarts -ge $maxRestarts) {
            Write-Host "[HB $ts] DEAD pkls=$pklCount/$totalRuns -- exceeded $maxRestarts restarts, GIVING UP"
            break
        }
        $restarts++
        $env:DEV_MODE = '1'
        $env:PYTHONIOENCODING = 'utf-8'
        Start-Process -FilePath python -ArgumentList 'run_task25.py' `
            -WorkingDirectory 'C:\Users\Admin\brain-organoid-rl' `
            -RedirectStandardOutput 'C:\Users\Admin\brain-organoid-rl\ablation_results\task25_run.log' `
            -RedirectStandardError  'C:\Users\Admin\brain-organoid-rl\ablation_results\task25_run.err' `
            -WindowStyle Hidden | Out-Null
        Write-Host "[HB $ts] DEAD pkls=$pklCount/$totalRuns -- RESTARTED (#$restarts)"
    }
    [Console]::Out.Flush()
}
