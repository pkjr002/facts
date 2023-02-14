#!/bin/bash

start=$(date +%s)

# Activate the vene
source ../ve3/bin/activate

# FOLDERS
EXP=('NZrerun.ssp585.7k')
EXP_FOLDER='experiments-pkjr002'
#
current_dir=$(pwd)

# LOG file
outpt="Tlog_$EXP_FOLDER.$EXP.txt"
echo -e "==> Started at: $(date --date=@$start)" > "$outpt"
echo -e "::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: \n" >> "$outpt"


# Loop over the experiments
for exp in "${EXP[@]}"; do
  #cd "$current_dir/$EXP_FOLDER/$exp"

  echo -e " ______________________________________________________________ " >> "$outpt"
  echo -e "==> begin scenario $exp ... " >> "$outpt"
  echo -e "experiment NAME:: $exp  |:|  experiment folder:: $EXP_FOLDER" >> "$outpt"
  #
  python3 runFACTS.py $EXP_FOLDER/$exp 2>&1 | tee -a $outpt
  #
  echo -e "\n ==> end \n" >> "$outpt"
  
  cd "$current_dir/$EXP_FOLDER/$exp"


done
cd "$current_dir"





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