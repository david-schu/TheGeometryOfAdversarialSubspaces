#!/bin/bash
#SBATCH --ntasks=1                      # Number of tasks (see below)
#SBATCH --cpus-per-task=5               # Number of CPU cores per task
#SBATCH --nodes=1                       # Ensure that all cores are on one machine
#SBATCH --time=0-01:00                  # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti          # Partition to submit to
#SBATCH --partition=cpu-short           # Force CPU node
#SBATCH --mem=500G                      # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=run_outputs/batch_curvature_%j.out # File to which STDOUT will be written
#SBATCH --error=run_outputs/batch_curvature_%j.err  # File to which STDERR will be written
#SBATCH --mail-type=END
#SBATCH --mail-user=dylan.paiton@uni-tuebingen.de

srun singularity exec --nv /mnt/qb/bethge/shared/dylan_david_shared/singularity/dpaiton_pytorch-2021-10-31-04d5c41e40c4.sif ./run_generate.sh

echo DONE!
