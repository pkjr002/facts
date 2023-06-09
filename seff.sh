#!/bin/bash

job_id=29034748
output_file="seff_output_Tlog.nzOG.ssp585.1kloc.2023-06-09-12-52-24-EDT.txt"
interval=3600  # 1 hour

get_seff_info() {
    echo "Current time: $(date)" >> "$output_file"
    echo "Job ID: $job_id" >> "$output_file"
    echo "------------------------" >> "$output_file"
    seff $job_id >> "$output_file"
    echo "--------xxx-------------" >> "$output_file"
    echo >> "$output_file"
}

# seff info @ reg intervals
while true; do
    get_seff_info
    sleep $interval
done
