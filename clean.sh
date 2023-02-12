#!/bin/bash

EXP=('NZrerun.ssp119' 'NZrerun.ssp126' 'NZrerun.ssp245')
EXP_FOLDER='experiments-pkjr002'
#
current_dir=$(pwd)

# SANDBOX files.
#[ -d /scratch/pk695/radical.pilot.sandbox ] && rm -rf /scratch/pk695/radical.pilot.sandbox/*


# FACTS ./ files
#[ -d /scratch/pk695/FACTS/002_fork/facts/re.session.* ] && rm -rf /scratch/pk695/FACTS/002_fork/facts/re.session.*


for exp in "${EXP[@]}"; do
  cd "$current_dir/$EXP_FOLDER/$exp"

  # Remove the generated FACTS experiment output files
  [ $(ls *output* 2> /dev/null) ] && rm -rf *output*
  #
  [ -e workflows.yml ] && rm -rf workflows.yml
  [ -e location.lst ] && rm -rf location.lst

done
cd "$current_dir"