$log = 'C:\Users\Admin\serial_position_experiment\results\phase3_resume.log'
$seen = 0
while ($true) {
    if (Test-Path $log) {
        $lines = Get-Content $log -ErrorAction SilentlyContinue
        if ($lines -and $lines.Count -gt $seen) {
            $newlines = $lines[$seen..($lines.Count - 1)]
            foreach ($line in $newlines) {
                if ($line -match '\[\d+/20\]|ALL PHASES COMPLETE|Error|Traceback|Killed|OOM|failed') {
                    Write-Output $line
                }
            }
            $seen = $lines.Count
        }
    }
    $proc = Get-Process -Id 14620 -ErrorAction SilentlyContinue
    if (-not $proc) {
        Write-Output "WARNING: PID 14620 is gone"
    }
    Start-Sleep -Seconds 30
}
