#!/bin/bash
#SBATCH -J gnn_sweep_50pct
#SBATCH -D  ./
#SBATCH -o /dss/dsshome1/0E/ge65koy2/logs/%x.%j.out
#SBATCH -e /dss/dsshome1/0E/ge65koy2/logs/%x.%j.err
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hoda.hamdy@tum.de


module load slurm_setup
source activate fleetpy
python studies/manhattan_case_study/run_gnn_demand_sweep_cluster.py
