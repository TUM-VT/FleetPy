#!/bin/bash
#SBATCH -J gnn_peak8day
#SBATCH -D  ./
#SBATCH -o /dss/dsshome1/0E/ge65koy2/logs/%x.%j.out
#SBATCH -e /dss/dsshome1/0E/ge65koy2/logs/%x.%j.err
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hoda.hamdy@tum.de


module load slurm_setup
source activate fleetpy
python gnn_project/run_experiment.py
