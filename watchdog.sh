#!/bin/bash
PYTHON="C:/Users/Admin/AppData/Local/Programs/Python/Python313/python.exe"
RESULTS="mod_results/mod3_learned_schema_15seeds.csv"
LOG="mod_results/mod3_resume.log"
TOTAL=70
BEAT=0

while true; do
    BEAT=$((BEAT+1))
    NOW=$(date +"%H:%M:%S")

    # Count rows
    if [ -f "$RESULTS" ]; then
        ROWS=$(tail -n +2 "$RESULTS" | wc -l)
    else
        ROWS=0
    fi
    LEFT=$((TOTAL - ROWS))
    ETA_H=$(( (LEFT * 900) / 3600 ))
    ETA_M=$(( ((LEFT * 900) % 3600) / 60 ))

    # Last log line
    if [ -f "$LOG" ]; then
        LAST=$(grep -v '^$' "$LOG" | tail -1)
        LOG_AGE=$(( $(date +%s) - $(date -r "$LOG" +%s 2>/dev/null || echo $(date +%s)) ))
    else
        LAST="(no log)"
        LOG_AGE=0
    fi

    # Kill zombies (only non-ancestor python run_experiment processes)
    ZOMBIES=""
    MY_PID=$$
    for PID in $(tasklist.exe 2>/dev/null | grep -i python | awk '{print $2}'); do
        CMD=$(wmic process where "ProcessId=$PID" get CommandLine 2>/dev/null | grep -i "run_experiment")
        if [ -n "$CMD" ]; then
            taskkill.exe /F /PID $PID 2>/dev/null && ZOMBIES="$ZOMBIES $PID"
        fi
    done

    echo "==============================="
    echo "BEAT #$BEAT @ $NOW"
    echo "  Rows: $ROWS/$TOTAL | ETA: ~${ETA_H}h ${ETA_M}m"
    echo "  Current: $LAST"
    echo "  Log age: ${LOG_AGE}s"
    if [ -n "$ZOMBIES" ]; then
        echo "  ZOMBIE KILLED: PID$ZOMBIES"
    fi
    echo "==============================="

    sleep 300
done
