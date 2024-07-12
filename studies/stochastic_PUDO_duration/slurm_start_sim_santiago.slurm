#!/bin/bash
#SBATCH -J FirstClusterTest
#SBATCH -o /gpfs/scratch/t5621/ge75mum2/ge75mum2/logs/first_test_logs.txt
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=cm2
#SBATCH --partition=cm2_tiny
#SBATCH --nodes=1
#SBATCH --tasks-per-node=28
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=end
#SBATCH --mail-user=santiago.alvarez@tum.de
#SBATCH --export=NONE
#SBATCH --time=00:01:00
#SBATCH --array=0-0

module load slurm_setup
module load gurobi
module load python
source activate fp_env

srun python slurm_run_job_array.py FleetPy/studies/stochastic_PUDO_duration/scenarios/constant_config_stochastic_PUDO.csv FleetPy/studies/stochastic_PUDO_duration/scenarios/scenario_config_stochastic_PUDO_DEBUGGING.csv
