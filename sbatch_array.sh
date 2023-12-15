#!/bin/bash
#SBATCH --array=0-2
#SBATCH --exclusive

source ~/anaconda3/etc/profile.d/conda.sh
conda activate qml

pwd
python_version="$(python --version)"
required_version="Python 3.9.12"
echo current python version is $python_version
echo required python version is $required_version

if [ "$python_version" = "$required_version" ]; then
    echo "Python 3.9.12 detected"
    echo $SLURM_ARRAY_TASK_ID
    python a.py --rnd_seed $SLURM_ARRAY_TASK_ID
else
    echo "Python 3.9.12 not detected"
fi