#!/bin/bash

EXPERIMENT="dummy"
# EXPERIMENT="coupling.fair"
# EXPERIMENT="coupling.ssp585.wfF.global"
# EXPERIMENT="coupling.ssp585.wfF"
# EXPERIMENT="coupling.ssp585"


EXPPATH="exp.alt.emis/$EXPERIMENT"
LOGFILE="${EXPPATH}/time_${EXPERIMENT}_$(whoami).log"

# Start timestamp
{
    echo "===== START: $(date '+%Y-%m-%d %H:%M:%S') ====="
    
    START_TIME=$(date +%s)

    # Run FACTS
    python runFACTS.py "$EXPPATH"

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    # Format duration as HH:MM:SS
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))
    FORMATTED_DURATION=$(printf "%02d:%02d:%02d" $HOURS $MINUTES $SECONDS)

    echo "===== END: $(date '+%Y-%m-%d %H:%M:%S') ====="
    echo "DURATION: $FORMATTED_DURATION (HH:MM:SS)"

} 2>&1 | tee "$LOGFILE"


# SCRATCH
# { time python runFACTS.py "$EXPPATH" ; } 2> "$LOGFILE"
# { time python runFACTS.py "$EXPPATH" ; } 2>&1 | tee "$LOGFILE"