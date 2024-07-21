#!/bin/bash

#SBATCH -J solar  # job name
#SBATCH -o sbatch_output_log/output_%x_%j.out # standard output and error log
#SBATCH -p RTX6000ADA  # queue name or partiton name
#SBATCH -t 72:00:00 # Run time (hh:mm:ss)
#SBATCH  --gres=gpu:2
#SBATCH  --nodes=1
#SBATCH  --ntasks=2
#SBATCH  --cpus-per-task=4
srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

module purge 
ml load ohpc

date
bash /home1/deokhk_1/project/mobis_PLC/train_solar_inst.sh

date
