#!/bin/bash
#SBATCH --job-name=decomp_batch
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=5         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=3-00:00            # Runtime in D-HH:MM
#SBATCH --partition=cpu-long    # Partition to submit to
#SBATCH --mem=100G              # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=cnn_%j.out  	  # File to which STDOUT will be written
#SBATCH --error=cnn_%j.err      # File to which STDERR will be written
##SBATCH --gres=gpu:5              # Request one GPU

#SBATCH --mail-type=END
#SBATCH --mail-user=david.schultheiss@student.uni-tuebingen.de


singularity exec --nv docker://davidschultheiss/ad:latest /opt/conda/bin/python3 decomp_fun.py $arg1 $arg2 $arg3

echo DONE!
