#!/bin/bash

start=$(date +%s)

# Activate the vene
source ../ve3/bin/activate

# FOLDERS
EXP=('NZrerun.ssp119')
EXP_FOLDER='experiments-pkjr002'
#
current_dir=$(pwd)

# LOG file
outpt="Tlog_$EXP_FOLDER.$EXP.txt"
echo -e "==> Started at: $(date --date=@$start)" > "$outpt"
echo -e "::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: \n" >> "$outpt"


# Loop over the experiments
for exp in "${EXP[@]}"; do
  cd "$current_dir/$EXP_FOLDER/$exp"

  # Large location file
  source_file="latlon_basedON_2kmNZInsar_IPdata_.txt"
  
  lines_per_iteration=500
  total_lines=7435
  # Counter for the number of lines copied so far
  lines_copied=0
  
  # Loop until all lines are copied
  for ((i=0; i<total_lines; i+=lines_per_iteration)); do
    
    # Calculate the number of lines to copy this iteration
    lines_to_copy=$((lines_per_iteration<total_lines-i?lines_per_iteration:total_lines-i))
  
    # Define the target file for this iteration
    #target_file="target_$((i/lines_per_iteration)).txt"
    target_file="location.lst"
  
    # Copy the lines
    head -n $((i+lines_to_copy)) "$source_file" | tail -n $lines_to_copy > "$target_file"
  
    # Update the lines_copied counter
    lines_copied=$((lines_copied+lines_to_copy))
  
    cd "$current_dir"

    echo -e " ______________________________________________________________ " >> "$outpt"
    echo -e "==> begin loop $i ... " >> "$outpt"
    echo -e "experiment NAME:: $exp  |:|  experiment folder:: $EXP_FOLDER" >> "$outpt"
    echo -e "location.lst:: lines_to_copy=$lines_to_copy lines_copied=$lines_copied \n" >> "$outpt"
    #
    python3 runFACTS.py $EXP_FOLDER/$exp 2>&1 | tee -a $output
    #
    echo -e "\n ==> end \n" >> "$outpt"
    
    # Rename op
    cd "$current_dir/$EXP_FOLDER/$exp"
    output_set="output_set_$((i/lines_per_iteration))"
    mv output/ "$output_set"
    rm -rf output/
    
  
    #rm -rf re.session*
  
  
  
  done


done
cd "$current_dir"





# ====================================================
# # Activate the vene
# source ../ve3/bin/activate
# #
# #
# # Experiment
# EXP='NZrerun.OG.FULL'
# EXP_FOLDER='experiments-pkjr002'
# #
# # RUN facts.
# python3 runFACTS.py $EXP_FOLDER/$EXP/ 2>&1 | tee $EXP_FOLDER/$EXP/OP-TERM-$EXP.txt




# ====================================================
# DO NOT USE Unless checked path
##rm -r re.session.amarel1.amarel.rutgers.edu.pk695.*
##rm -r /scratch/pk695/radical.pilot.sandbox/re.session.amarel1.amarel.rutgers.edu.pk695.019393.0003*




echo -e "\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::" >> "$outpt"
echo "!! End of Program !!" >> "$outpt"
end=$(date +%s)
# Calculate the duration in seconds
duration=$((end - start))
duration_min=$((duration / 60))
#
# Print the start time, end time, and duration to a file

echo -e "\nEXPERIMENT FOLDER:: "$EXP_FOLDER >> "$outpt"
echo "EXPERIMENT:: "$exp >> "$outpt"
echo " ............... " >> "$outpt"
echo "Started at: $(date --date=@$start)" >> "$outpt"
echo "Ended at: $(date --date=@$end)" >> "$outpt"
echo "Duration: $duration_min minutes" >> "$outpt"