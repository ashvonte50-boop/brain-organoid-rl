$script = "C:\Users\Admin\brain-organoid-rl\mod_results\mod3_learned_schema_15seeds.py"
$csv   = "C:\Users\Admin\brain-organoid-rl\mod_results\mod3_learned_schema_15seeds.csv"
$log   = "C:\Users\Admin\brain-organoid-rl\mod_results\mod3_watchdog.log"
$python = "python"

function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line -ForegroundColor Cyan
    Add-Content $log $line
}

Log "Watchdog started. Monitoring MOD-3."

while ($true) {
    # Count completed rows
    $rows = 0
    if (Test-Path $csv) {
        $rows = (Get-Content $csv | Measure-Object -Line).Lines - 1
    }

    if ($rows -ge 70) {
        Log "MOD-3 COMPLETE — 70/70 rows in CSV. Watchdog exiting."
        break
    }

    Log "Starting MOD-3 (CSV rows so far: $rows/70)..."
    $proc = Start-Process $python -ArgumentList $script -PassThru -NoNewWindow
    Log "Process started: PID $($proc.Id)"

    # Monitor: check every 60s if process is alive and CSV is growing
    $lastRows = $rows
    $stuckCount = 0

    while ($true) {
        Start-Sleep -Seconds 60

        $proc.Refresh()
        if ($proc.HasExited) {
            Log "Process exited (code $($proc.ExitCode)). Checking CSV..."
            break
        }

        # Check if CSV is growing (not stuck)
        $newRows = 0
        if (Test-Path $csv) {
            $newRows = (Get-Content $csv | Measure-Object -Line).Lines - 1
        }

        if ($newRows -eq $lastRows) {
            $stuckCount++
            Log "No new rows for $stuckCount min (rows=$newRows). Each run takes ~15-30min so waiting..."
            if ($stuckCount -ge 40) {
                Log "STUCK 40 min with no new rows — killing and restarting."
                Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
                break
            }
        } else {
            $stuckCount = 0
            Log "Progress: $newRows/70 rows (+$($newRows - $lastRows) new)"
            $lastRows = $newRows
        }
    }

    $finalRows = 0
    if (Test-Path $csv) {
        $finalRows = (Get-Content $csv | Measure-Object -Line).Lines - 1
    }

    if ($finalRows -ge 70) {
        Log "MOD-3 COMPLETE — 70/70 rows. Watchdog done."
        break
    }

    Log "Restarting in 10 seconds... (rows so far: $finalRows/70)"
    Start-Sleep -Seconds 10
}

Log "Watchdog finished."
