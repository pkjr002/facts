  #!/bin/bash

  # Configuration - SLURM-aware
  if [ ! -z "$SLURM_JOB_ID" ]; then
      LOGFILE="${SLURM_SUBMIT_DIR}/process_monitor_${SLURM_JOB_ID}.log"
      echo "SLURM Job ID: $SLURM_JOB_ID" > "$LOGFILE"
      echo "Node: $SLURMD_NODENAME" >> "$LOGFILE"
  else
      LOGFILE="process_monitor_coupling.ssp245.log"
  fi

  # Add signal handling (including SLURM signals)
  cleanup() {
      echo "Cleaning up..."
      if [ ! -z "$MONITOR_PID" ]; then
          kill $MONITOR_PID 2>/dev/null
      fi
      exit
  }

  trap cleanup SIGTERM SIGINT SIGQUIT SIGUSR1 SIGUSR2

# Start monitoring with actual code
(
while true; do
    echo "=== $(date +'%Y-%m-%d %H:%M:%S') ===" >> "$LOGFILE"
    ps -u $USER -o pid,ppid,%cpu,%mem,vsz,rss,psr,command |
grep python | \
    awk '{printf "%5s %5s %5s %5s %8.2fGB %8.2fGB %3s ", $1, $2, $3, $4, $5/1048576, $6/1048576, $7; for(i=8;i<=NF;i++) printf "%s ", $i; print ""}' >> "$LOGFILE"
    sleep $((60*4))
done
) &
MONITOR_PID=$!

python runFACTS.py exp.alt.emis/coupling.ssp245/

# Normal cleanup
cleanup