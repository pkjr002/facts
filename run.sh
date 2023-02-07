#!/bin/bash


# Activate the vene
source ../ve3/bin/activate
#
#
# Experiment
EXP='NZrerun.OG.FULL'
EXP_FOLDER='experiments-pkjr002'
#
# RUN facts.
python3 runFACTS.py $EXP_FOLDER/$EXP/ 2>&1 | tee $EXP_FOLDER/$EXP/OP-TERM-$EXP.txt





# DO NOT USE Unless checked path
##rm -r re.session.amarel1.amarel.rutgers.edu.pk695.*
##rm -r /scratch/pk695/radical.pilot.sandbox/re.session.amarel1.amarel.rutgers.edu.pk695.019393.0003*