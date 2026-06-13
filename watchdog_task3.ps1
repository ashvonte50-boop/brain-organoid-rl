# Watchdog for Task 3
$restarts = 0; $maxRestarts = 5; $totalRuns = 10
while ($true) {
    Start-Sleep -Seconds 480
    $py = Get-CimInstance Win32_Process |
        Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*run_task3*' } |
        Select-Object -First 1
    $pklCount = @(Get-ChildItem 'C:\Users\Admin\brain-organoid-rl\ablation_results\task3\T3_*.pkl' -ErrorAction SilentlyContinue).Count
    $lastFile = Get-ChildItem 'C:\Users\Admin\brain-organoid-rl\ablation_results\task3\' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $minAgo = if ($lastFile) { [Math]::Round(((Get-Date)-$lastFile.LastWriteTime).TotalMinutes,1) } else { -1 }
    $ts = Get-Date -Format 'HH:mm'
    if ($py) {
        Write-Host "[HB $ts] ALIVE pid=$($py.ProcessId) pkls=$pklCount/$totalRuns last_write=${minAgo}min_ago"
    } elseif ($pklCount -ge $totalRuns) {
        Write-Host "[HB $ts] DONE pkls=$pklCount/$totalRuns watchdog exiting"; break
    } else {
        if ($restarts -ge $maxRestarts) { Write-Host "[HB $ts] DEAD - max restarts reached"; break }
        $restarts++
        $env:DEV_MODE = '1'; $env:PYTHONIOENCODING = 'utf-8'
        Start-Process -FilePath python -ArgumentList 'run_task3.py' `
            -WorkingDirectory 'C:\Users\Admin\brain-organoid-rl' `
            -RedirectStandardOutput 'C:\Users\Admin\brain-organoid-rl\ablation_results\task3_run.log' `
            -RedirectStandardError  'C:\Users\Admin\brain-organoid-rl\ablation_results\task3_run.err' `
            -WindowStyle Hidden | Out-Null
        Write-Host "[HB $ts] DEAD pkls=$pklCount/$totalRuns -- RESTARTED (#$restarts)"
    }
    [Console]::Out.Flush()
}
