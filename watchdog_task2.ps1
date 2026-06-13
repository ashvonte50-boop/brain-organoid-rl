# Watchdog for Task 2 long-running sweep.
# Polls every 480s (8 min). Emits ONE heartbeat line per check.
# Auto-restarts the parent run_task2.py if dead (cached pkls are skipped on resume).

$restarts = 0
$maxRestarts = 5

while ($true) {
    Start-Sleep -Seconds 480

    $py = Get-CimInstance Win32_Process |
        Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*run_task2*' } |
        Select-Object -First 1

    $pklCount = @(Get-ChildItem 'C:\Users\Admin\brain-organoid-rl\ablation_results\task2\T2_*.pkl' -ErrorAction SilentlyContinue).Count

    $lastFile = Get-ChildItem 'C:\Users\Admin\brain-organoid-rl\ablation_results\task2\' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $minAgo = if ($lastFile) { [Math]::Round(((Get-Date) - $lastFile.LastWriteTime).TotalMinutes, 1) } else { -1 }

    $ts = Get-Date -Format 'HH:mm'

    if ($py) {
        Write-Host "[HB $ts] ALIVE pid=$($py.ProcessId) pkls=$pklCount/40 last_write=${minAgo}min_ago"
    } elseif ($pklCount -ge 40) {
        Write-Host "[HB $ts] DONE pkls=40/40 watchdog exiting"
        break
    } else {
        if ($restarts -ge $maxRestarts) {
            Write-Host "[HB $ts] DEAD pkls=$pklCount/40 -- exceeded $maxRestarts restarts, GIVING UP"
            break
        }
        $restarts++
        $env:DEV_MODE = '1'
        $env:PYTHONIOENCODING = 'utf-8'
        Start-Process -FilePath python -ArgumentList 'run_task2.py' `
            -WorkingDirectory 'C:\Users\Admin\brain-organoid-rl' `
            -RedirectStandardOutput 'C:\Users\Admin\brain-organoid-rl\ablation_results\task2_run.log' `
            -RedirectStandardError  'C:\Users\Admin\brain-organoid-rl\ablation_results\task2_run.err' `
            -WindowStyle Hidden | Out-Null
        Write-Host "[HB $ts] DEAD pkls=$pklCount/40 -- RESTARTED (#$restarts)"
    }
    [Console]::Out.Flush()
}
