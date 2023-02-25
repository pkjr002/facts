#!/bin/bash
# 
startOG=$(date +%s)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
FULL_LOCATION_LOOP=1     #[1-yes,     0-do it in increments; values need to be set in the 2 lines below]
#
total_lines=66190  #location.lst length
lines_per_iteration=500  #location chunk size
# .....................................................................................................................
venv="memfix"
source ../$venv/bin/activate   # Activate the vene
# .....................................................................................................................
facts_dir=$(pwd)
#
EXP_FOLDER="experiments-pkjr002";   EXP_MAIN="FACTS_test";     EXP=("coupling.ssp585")
#echo "EXP_FOLDER=$EXP_FOLDER        EXP_MAIN=$EXP_MAIN      EXP=$EXP"
#
experiment_dir="$facts_dir/$EXP_FOLDER/$EXP_MAIN/$EXP"
# .....................................................................................................................
fileOUT="Tlog_$EXP_MAIN.$EXP.txt"      #"target_$((i/lines_per_iteration)).txt"
#
source_file="FULL_location.lst"
#
target_file="location.lst"
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
cd "$experiment_dir"
#
# Create a LOG file for the Experiment.
truncate -s 0 "$facts_dir/$fileOUT"
echo -e "!!! Start of Program !!! $(date --date=@$startOG)" 2>&1 | tee "$facts_dir/$fileOUT"
echo -e "venv --> $venv" 2>&1 | tee -a "$facts_dir/$fileOUT"
echo -e "::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: \n" 2>&1 | tee -a "$facts_dir/$fileOUT"
#
# Loop over the experiments (Have to make more efficient before using)
for exp in "${EXP[@]}"; do
  #
  # For large location.list, split it and run in increments.  
  if [[ "$FULL_LOCATION_LOOP" -eq 0 ]]; then
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
      echo -e " ___________________________________________________________________________________________ " 2>&1 | tee -a "$facts_dir/$fileOUT"
      current_loop_NO="$(($i/$lines_per_iteration))"
      Total_Loop_NO=$(echo "scale=0; ($total_lines/$lines_per_iteration)+1" | bc -l)
      echo -e "==> begin loop $current_loop_NO of $Total_Loop_NO ..." 2>&1 | tee -a "$facts_dir/$fileOUT"
      start=$(date +%s)
      echo "Started at:: $(date --date=@$start)" 2>&1 | tee -a "$facts_dir/$fileOUT"
      echo -e "exp NAME:: $exp  |:|  exp sub-FOLDER:: $EXP_MAIN |:|  exp FOLDER:: $EXP_FOLDER" 2>&1 | tee -a "$facts_dir/$fileOUT"
      echo -e "Source Location File:: $source_file |:| Temp Location File $target_file" 2>&1 | tee -a "$facts_dir/$fileOUT"
      echo -e "Chunk:: line $i T line $lines_copied" 2>&1 | tee -a "$facts_dir/$fileOUT"
      echo -e " ___________________________________________________________________________________________ \n" 2>&1 | tee -a "$facts_dir/$fileOUT"
      #
      cd $facts_dir
      python3 runFACTS.py $EXP_FOLDER/$EXP_MAIN/$exp 2>&1 | tee -a "$facts_dir/$fileOUT"
      cd "$experiment_dir"
      #
      end=$(date +%s)
      echo -e "\n Ended at:: $(date --date=@$end)" 2>&1 | tee -a "$facts_dir/$fileOUT"
      echo -e "\n ==> end \n" 2>&1 | tee -a "$facts_dir/$fileOUT"
      # Rename output file to allow for next set of outputs. 
      output_set="output_set_$((i/lines_per_iteration))"
      mv output/ "$output_set"
      rm -rf output/  
    done  
  else # Only use If you have <500 locations.
    echo -e " ___________________________________________________________________________________________ " 2>&1 | tee -a "$facts_dir/$fileOUT"
    echo -e "==> begin (single loop) ... " 2>&1 | tee -a "$facts_dir/$fileOUT"
    start=$(date +%s)
    echo "Started at:: $(date --date=@$start)" 2>&1 | tee -a "$facts_dir/$fileOUT"
    echo -e "exp NAME:: $exp  |:|  exp sub-FOLDER:: $EXP_MAIN |:|  exp FOLDER:: $EXP_FOLDER" 2>&1 | tee -a "$facts_dir/$fileOUT"
    echo -e " ___________________________________________________________________________________________ \n" 2>&1 | tee -a "$facts_dir/$fileOUT"
    #
    cd "$facts_dir"
    python3 runFACTS.py $EXP_FOLDER/$EXP_MAIN/$exp 2>&1 | tee -a $facts_dir/$fileOUT
    cd $experiment_dir
    #
    end=$(date +%s)
    echo -e "\n Ended at:: $(date --date=@$end)" 2>&1 | tee -a "$facts_dir/$fileOUT"
    echo -e "\n ==> end \n" 2>&1 | tee -a "$facts_dir/$fileOUT"
  fi
done
cd "$facts_dir"
#
# Write the footer.
echo -e "\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::" 2>&1 | tee -a "$facts_dir/$fileOUT"
echo "!!! End of Program !!!" 2>&1 | tee -a "$facts_dir/$fileOUT"
endOG=$(date +%s)
# Calculate the duration in seconds
duration=$((endOG - startOG))
duration_min=$((duration / 60))
# Print the start time, end time, and duration to a file
echo -e "\nEXPERIMENT FOLDER:: "$EXP_FOLDER 2>&1 | tee -a "$facts_dir/$fileOUT"
echo -e "EXPERIMENT subFOLDER:: "$EXP_MAIN 2>&1 | tee -a "$facts_dir/$fileOUT"
echo "EXPERIMENT:: "$exp 2>&1 | tee -a "$facts_dir/$fileOUT"
echo -e "venv --> $venv" 2>&1 | tee -a "$facts_dir/$fileOUT"
echo " ............... " 2>&1 | tee -a "$facts_dir/$fileOUT"
echo "Started at:: $(date --date=@$startOG)" 2>&1 | tee -a "$facts_dir/$fileOUT"
echo "Ended at:: $(date --date=@$endOG)" 2>&1 | tee -a "$facts_dir/$fileOUT"
echo "Duration:: $duration_min minutes" 2>&1 | tee -a "$facts_dir/$fileOUT"






# ====================================================
# DO NOT USE Unless checked path
##rm -r re.session.amarel1.amarel.rutgers.edu.pk695.*
##rm -r /scratch/pk695/radical.pilot.sandbox/re.session.amarel1.amarel.rutgers.edu.pk695.019393.0003*