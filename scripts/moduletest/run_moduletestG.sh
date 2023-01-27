#!/bin/bash
#SBATCH --partition=main
#SBATCH --mem=128000
#SBATCH --time=12:00:00

# Activate the conda environment. Replace env by your environment (e.g. base)
env="facts37" ; source ~/.bashrc ;conda activate $env

# FACTS main Directory
cd ../../
FACTS_DIR=`pwd`

# Jump to Modules test Directory. 
#cd $FACTS_DIR/modules/$MOD/$subMOD/test
cd $FACTS_DIR/modules/$MOD/$SUBMOD/test


# Append date-time to each run.
dt=`date +%s`

# Set directories for sandboxing each module.
TESTSCRIPT_DIR=`pwd`
FACTSSCRIPT_DIR=$TESTSCRIPT_DIR/../../../../scripts
#LOG_STDOUT=$TESTSCRIPT_DIR/test.out.$dt
LOG_STDOUT=$TESTSCRIPT_DIR/test.out
#LOG_STDERR=$TESTSCRIPT_DIR/test.err.$dt
LOG_STDERR=$TESTSCRIPT_DIR/test.err

# Run modules within sandbox folder.
source $FACTSSCRIPT_DIR/moduletest/moduletest.config.global
source moduletest.config
source $FACTSSCRIPT_DIR/moduletest/moduletest.sh > $LOG_STDOUT 2> $LOG_STDERR

yes '' | sed 2q 

echo "Launching test script. 
Logging output to :
$LOG_STDOUT 
$LOG_STDERR." 

yes '' | sed 2q

# Check to see if there are errors. 
[ -s $LOG_STDERR ] && echo "ERROR: Check test.err" || echo "Successful Run"

yes '' | sed 2q

# If no errors, remove err and out file.
if ! [ -s $LOG_STDERR ] ; then 
    rm $LOG_STDERR $LOG_STDOUT 
    echo "deleting .err and .out files" 
fi


#echo "\n $(pwd)" 
#cd $FACTSSCRIPT_DIR/moduletest/