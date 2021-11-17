#!/bin/bash
#SBATCH --ntasks=1                      # Number of tasks (see below)
#SBATCH --nodes=1                       # Ensure that all cores are on one machine
#SBATCH --time=2-00:00                  # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti          # Partition to submit to
#SBATCH --cpus-per-task=8               # Number of CPU cores per task
#SBATCH --mem=200G                      # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=run_outputs/batch_curvature_%j.out # File to which STDOUT will be written
#SBATCH --error=run_outputs/batch_curvature_%j.err  # File to which STDERR will be written
#SBATCH --gres=gpu:1                    # optionally type and number of gpus
#SBATCH --mail-type=END
#SBATCH --mail-user=dylan.paiton@uni-tuebingen.de

echo "parameters: $arg1 $arg2 $arg3"
srun singularity exec --nv /mnt/qb/bethge/shared/dylan_david_shared/singularity/dpaiton_pytorch_latest-2021-10-03-f617935c6553.sif /opt/conda/bin/python3 subspace_curvature.py $arg1 $arg2 $arg3

echo DONE!
