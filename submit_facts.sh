#!/bin/bash
#SBATCH --partition=kopp_1        # Partition (job queue)
#SBATCH --job-name=facts-2node    # Assign a short name to your job
#SBATCH --nodes=1                 # Number of nodes you require
#SBATCH --ntasks-per-node=24      # Total # of tasks across all nodes
#SBATCH --time=03:30:00           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.%N.%j.out  # STDOUT output file
#SBATCH --error=slurm.%N.%j.err   # STDERR output file (optional)
radical-stack

export RADICAL_LOG_LVL="DEBUG"
export RADICAL_PROFILE="TRUE"

python3 /scratch/pk695/FACTS/2023_12_GISS/facts/runFACTS.py /scratch/pk695/FACTS/2023_12_GISS/facts/$1