"""
Standalone heartbeat monitor — shows Windows toast notification every 2 min.
Run this in a separate terminal: python heartbeat_monitor.py
"""
import time, os, subprocess, sys

CSV = r'C:\Users\Admin\brain-organoid-rl\major1_results\major1_decoupling.csv'
LOG = r'C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin-brain-organoid-rl\9903e6bf-dd2d-4bc9-9b2a-88bc229d9208\tasks\bvj6m0loh.output'
INTERVAL = 120  # 2 minutes

def count_rows(path):
    try:
        with open(path, 'r') as f:
            return max(0, sum(1 for _ in f) - 1)
    except:
        return 0

def get_last_log_line(path):
    try:
        with open(path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        return lines[-1] if lines else "no output"
    except:
        return "log not readable"

def windows_notify(title, msg):
    # PowerShell toast notification
    ps = f'''
Add-Type -AssemblyName System.Windows.Forms
$n = New-Object System.Windows.Forms.NotifyIcon
$n.Icon = [System.Drawing.SystemIcons]::Information
$n.BalloonTipTitle = "{title}"
$n.BalloonTipText = "{msg}"
$n.Visible = $true
$n.ShowBalloonTip(8000)
Start-Sleep -Seconds 9
$n.Dispose()
'''
    subprocess.Popen(['powershell', '-WindowStyle', 'Hidden', '-Command', ps],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def beep():
    subprocess.Popen(['powershell', '-Command', '[Console]::Beep(1000,300)'],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("HEARTBEAT MONITOR STARTED — updates every 2 min", flush=True)
print("Press Ctrl+C to stop\n", flush=True)

while True:
    rows = count_rows(CSV)
    last = get_last_log_line(LOG)
    total = 45
    pct = int(100 * rows / total)

    msg = f"MAJOR-1: {rows}/{total} ({pct}%) | {last[:80]}"
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    title = f"MAJOR-1 Progress: {rows}/{total} done"
    windows_notify(title, msg)
    beep()

    time.sleep(INTERVAL)
