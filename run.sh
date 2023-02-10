#!/bin/bash

current_dir=$(pwd)

# Activate the vene
source ../ve3/bin/activate


# Experiment
EXP='NZrerun.ssp585'
EXP_FOLDER='experiments-pkjr002'
cd "$EXP_FOLDER/$EXP"

source_file="latlon_basedON_2kmNZInsar_IPdata_.txt"

lines_per_iteration=1000
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
  python3 runFACTS.py $EXP_FOLDER/$EXP/ 
  
  # Rename op
  cd "$EXP_FOLDER/$EXP"
  output_set="output_$((i/lines_per_iteration)).txt"
  mv -r output/ "$output_set"
  rm -rf output/
  
  cd "$current_dir"

  rm -rf re.session*



done



cd "$current_dir"


# RUN facts.
# python3 runFACTS.py $EXP_FOLDER/$EXP/ 




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