#!/bin/bash

#---------------------------------------------------
# SELECT Experiment FOLDER
EXP="experiments"

#---------------------------------------------------
# SELECT Experiment 
#
# EXPERIMENT="dummy"
# EXPERIMENT="coupling.ssp126"
# EXPERIMENT="coupling.ssp245"
# EXPERIMENT="coupling.ssp585"

# EXPERIMENT="srcHL"

EXPERIMENT="src.H.ssp370"
# EXPERIMENT="src.HL.ssp585"
# EXPERIMENT="src.L.ssp245"
# EXPERIMENT="src.LN.ssp245"

# EXPERIMENT="src.M.ssp245"
# EXPERIMENT="src.ML.ssp245"
# EXPERIMENT="src.VL.ssp126"

#---------------------------------------------------
# Change the timezone
TZVAR="America/New_York"
#---------------------------------------------------
EXPPATH="$EXP/$EXPERIMENT"
LOGFILE="${EXPPATH}/time_${EXPERIMENT}_$(whoami).log"
#---------------------------------------------------
# Start timestamp
{
    echo "===== START: $(TZ="$TZVAR" date '+%Y-%m-%d %H:%M:%S') EST ====="
    
    START_TIME=$(TZ="$TZVAR" date +%s)

    # Run FACTS
    python runFACTS.py "$EXPPATH"

    END_TIME=$(TZ="$TZVAR" date +%s)
    DURATION=$((END_TIME - START_TIME))

    # Format duration as HH:MM:SS
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))
    FORMATTED_DURATION=$(printf "%02d:%02d:%02d" $HOURS $MINUTES $SECONDS)

    echo "===== END: $(TZ="$TZVAR" date '+%Y-%m-%d %H:%M:%S') ====="
    echo "DURATION: $FORMATTED_DURATION (HH:MM:SS)"

} 2>&1 | tee "$LOGFILE"