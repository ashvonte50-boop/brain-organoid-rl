"""
Detached production launcher — avoids Windows multiprocessing + I/O redirect crash.
Spawns compare_catastrophic_forgetting.py as a fully detached process.
"""
import subprocess, sys, os, pathlib

here = pathlib.Path(__file__).parent
log  = here / "prod_run3.log"
err  = here / "prod_run3_err.log"

DETACHED_PROCESS      = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

import sys as _sys
script = _sys.argv[1] if len(_sys.argv) > 1 else "compare_catastrophic_forgetting.py"
log  = here / (script.replace(".py", "") + "_out.log")
err  = here / (script.replace(".py", "") + "_err.log")

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

p = subprocess.Popen(
    [sys.executable, "-u", script],
    stdout=open(log, "w"),
    stderr=open(err, "w"),
    cwd=str(here),
    env=env,
    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
)
print(f"Launched PID {p.pid}")
print(f"stdout -> {log}")
print(f"stderr -> {err}")
