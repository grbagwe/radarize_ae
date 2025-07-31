#!/bin/bash

# Submit this script with: sbatch FILENAME

#SBATCH --job-name=radarize_eval               # Name of the job
#SBATCH --output=slurm_data/radarize_eval_%j.out           # Standard output (%j = Job ID)
#SBATCH --error=slurm_data/radarize_eval_%j.err            # Standard error (%j = Job ID)
#SBATCH --time=08:00:00                    # Max runtime (HH:MM:SS)
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --cpus-per-task=32                  # Number of CPU cores per task
#SBATCH --mem 64gb           # amount of memory per CPU core (Memory per Task / Cores per Task)
# Created with the RCD Docs Job Builder
#
# Visit the following link to edit this job:
# https://docs.rcd.clemson.edu/palmetto/job_management/job_builder/?num_mem=12gb&use_gpus=yes&gpu_model=v100s&walltime=2%3A00%3A00&account_name=gbagwe&job_name=radarize_eval

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
# Load necessary modules
module load anaconda3
source activate radarize_ae # this is the env name you use

# cd your/project/dir #just to make sure that it can find the code # use the abs directory path 

cd /scratch/gbagwe/Projects/radar_slam/radarize_ae

# Run your code
./run_eval.sh
