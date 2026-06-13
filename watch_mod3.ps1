$csv = "C:\Users\Admin\brain-organoid-rl\mod_results\mod3_learned_schema_15seeds.csv"
$log = "C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin-brain-organoid-rl\80daba29-76a7-4002-a3fa-c379d7427bed\tasks\b1n3olwkz.output"

Write-Host "Watching MOD-3... Ctrl+C to stop" -ForegroundColor Cyan

while ($true) {
    $time = Get-Date -Format "HH:mm:ss"
    $rows = (Get-Content $csv | Measure-Object -Line).Lines - 1  # subtract header
    $lastLog = (Get-Content $log -Tail 3) -join " | "
    Write-Host "[$time] CSV rows: $rows/70 | $lastLog" -ForegroundColor Green
    Start-Sleep -Seconds 120
}
