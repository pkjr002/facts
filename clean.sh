#!/bin/bash

EXP=('NZrerun.ssp119')
EXP_FOLDER='experiments-pkjr002'
#
current_dir=$(pwd)


rm -r /scratch/pk695/radical.pilot.sandbox/*

for exp in "${EXP[@]}"; do
  cd "$current_dir/$EXP_FOLDER/$exp"

  rm -rf output/ workflows.yml location.lst
