#!/bin/bash
# 
start=$(date +%s)
# -------------------------------------------------------------------------------
FULLTG=1
total_lines=7435
lines_per_iteration=500
source_file="latlon_basedON_2kmNZInsar_IPdata_.txt"
target_file="location.lst"
# -------------------------------------------------------------------------------
# Activate the vene
source ../ve3/bin/activate
#
facts_dir=$(pwd)
# FOLDERS
EXP_FOLDER="experiments-pkjr002"
EXP_MAIN="NZrerun"
EXP=("NZrerun.ssp585.20k.50core")
experiment_dir="$facts_dir/$EXP_FOLDER/$EXP_MAIN/$EXP"
fileOUT="Tlog_$EXP_MAIN.$EXP.txt"      #"target_$((i/lines_per_iteration)).txt"
# -------------------------------------------------------------------------------
cd "$experiment_dir"
#
# Create a LOG file for the Experiment.
echo -e "==> Started at: $(date --date=@$start)" > "$facts_dir/$fileOUT"
echo -e "::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: \n" >> "$facts_dir/$fileOUT"
#
# Loop over the experiments (Have to make more efficient before using)
for exp in "${EXP[@]}"; do
  #
  # For large location.list, split it and run in increments.  
  if [[ "$FULLTG" -eq 1 ]]; then
    # Counter for the number of lines copied so far
    lines_copied=0
    # Loop until all lines are copied
    for ((i=0; i<total_lines; i+=lines_per_iteration)); do
      # Calculate the number of lines to copy this iteration
      lines_to_copy=$((lines_per_iteration<total_lines-i?lines_per_iteration:total_lines-i))
      # Copy the lines
      head -n $((i+lines_to_copy)) "$source_file" | tail -n $lines_to_copy > "$target_file"
      # Update the lines_copied counter
      lines_copied=$((lines_copied+lines_to_copy))
      #
      # Start logging the Experiment.   
      echo -e " ______________________________________________________________ " >> "$facts_dir/$fileOUT"
      echo -e "==> begin loop $i ... " >> "$facts_dir/$fileOUT"
      echo -e "experiment NAME:: $exp  |:|  experiment folder:: $EXP_FOLDER" >> "$facts_dir/$fileOUT"
      echo -e "location.lst:: lines_to_copy=$lines_to_copy lines_copied=$lines_copied \n" >> "$facts_dir/$fileOUT"
      #
      cd $facts_dir
      python3 runFACTS.py $EXP_FOLDER/$EXP_MAIN/$exp 2>&1 | tee -a $facts_dir/$fileOUT
      cd "$experiment_dir"
      #
      echo -e "\n ==> end \n" >> "$facts_dir/$fileOUT"
      # Rename output file to allow for next set of outputs. 
      output_set="output_set_$((i/lines_per_iteration))"
      mv output/ "$output_set"
      rm -rf output/  
    done  
  else # Only use If you have <500 locations.
    echo -e " ______________________________________________________________ " >> "$facts_dir/$fileOUT"
    echo -e "==> begin (single loop) ... " >> "$facts_dir/$fileOUT"
    echo -e "experiment NAME:: $exp  |:|  experiment folder:: $EXP_FOLDER" >> "$facts_dir/$fileOUT"
    #
    cd "$facts_dir"
    python3 runFACTS.py $EXP_FOLDER/$EXP_MAIN/$exp 2>&1 | tee -a $facts_dir/$fileOUT
    cd $experiment_dir
    #
    echo -e "\n ==> end \n" >> "$facts_dir/$fileOUT"
  fi
done
cd "$facts_dir"
#
# Write the footer.
echo -e "\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::" >> "$facts_dir/$fileOUT"
echo "!! End of Program !!" >> "$facts_dir/$fileOUT"
end=$(date +%s)
# Calculate the duration in seconds
duration=$((end - start))
duration_min=$((duration / 60))
# Print the start time, end time, and duration to a file
echo -e "\nEXPERIMENT FOLDER:: "$EXP_FOLDER >> "$facts_dir/$fileOUT"
echo "EXPERIMENT:: "$exp >> "$facts_dir/$fileOUT"
echo " ............... " >> "$facts_dir/$fileOUT"
echo "Started at: $(date --date=@$start)" >> "$facts_dir/$fileOUT"
echo "Ended at: $(date --date=@$end)" >> "$facts_dir/$fileOUT"
echo "Duration: $duration_min minutes" >> "$facts_dir/$fileOUT"






# ====================================================
# DO NOT USE Unless checked path
##rm -r re.session.amarel1.amarel.rutgers.edu.pk695.*
##rm -r /scratch/pk695/radical.pilot.sandbox/re.session.amarel1.amarel.rutgers.edu.pk695.019393.0003*