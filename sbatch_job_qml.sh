#!/bin/bash
#SBATCH --array=0-3
#SBATCH --exclusive

rnd_seed=0 # <---------------------------------------------------------------------
date_time="20230911_082815" # <---------------------------------------------------------------------

echo $date_time

echo
echo start running bash script and load conda python
echo

python_version="$(python --version)"
required_version="Python 3.9.12"
echo current python version is $python_version
echo required python version is $required_version
echo

if [ "$python_version" = "$required_version" ]; then
    echo "Python 3.9.12 detected"
    
    if [ $(expr $SLURM_ARRAY_TASK_ID % 4) -eq 0 ]; then
        q_gnn_layers=1
        q_gnn_reupload=0
    elif [ $(expr $SLURM_ARRAY_TASK_ID % 4) -eq 1 ]; then
        q_gnn_layers=2
        q_gnn_reupload=0
    elif [ $(expr $SLURM_ARRAY_TASK_ID % 4) -eq 2 ]; then
        q_gnn_layers=1
        q_gnn_reupload=1
    elif [ $(expr $SLURM_ARRAY_TASK_ID % 4) -eq 3 ]; then
        q_gnn_layers=2
        q_gnn_reupload=1
    fi
    echo running q_gnn_layers = $q_gnn_layers, q_gnn_reupload = $q_gnn_reupload, rnd_seed = $rnd_seed
    python a.py --date_time $date_time --q_gnn_layers $q_gnn_layers --q_gnn_reupload $q_gnn_reupload --rnd_seed $rnd_seed
else
    echo "Python 3.9.12 not detected"
fi
