#!/bin/bash
#SBATCH --job-name=dipS4          # create a short name for your job
#SBATCH --partition=matador
#SBATCH --nodes=1                # node count
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=20       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --array=0-19:1           # job array with index values
#%module load matador/0.15.4
#%module load gcc/9.3.0
#%module load cudnn/8.0.1.13-1.cuda11.0
#%module load cuda/11.0

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

. $HOME/conda/etc/profile.d/conda.sh
conda activate pytorch_gpu
# python super-resolution-rs.py --type Wind --factor 4 --slice $SLURM_ARRAY_TASK_ID --num_iter 6000 --process extract --savePath extract32_iter6k

python super-resolution-rs.py --type Solar --factor 4 --slice $SLURM_ARRAY_TASK_ID --num_iter 6000 --process extract --savePath extract32_iter6k