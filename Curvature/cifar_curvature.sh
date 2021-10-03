#!/bin/bash
#SBATCH --job-name=curvature_analysis
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=5         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-04:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --mem=100G                # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=curvature_%j.out # File to which STDOUT will be written
#SBATCH --error=curvature_%j.err  # File to which STDERR will be written
#SBATCH --gres=gpu:1              # optionally type and number of gpus

#SBATCH --mail-type=END
#SBATCH --mail-user=dylan.paiton@uni-tuebingen.de

singularity exec --nv docker://davidschultheiss/ad:latest /opt/conda/bin/python3 subspace_curvature.py $arg1 $arg2 $arg3

echo DONE!
